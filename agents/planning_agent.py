"""Planning agent for orchestrating multi-step pharmaceutical queries."""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agents.drug_agent import DrugAgent, DrugResponse
from agents.trial_agent import TrialAgent, TrialResponse
from config import config

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of pharmaceutical queries."""
    DRUG_INFO = "drug_info"
    DRUG_INTERACTION = "drug_interaction"
    CLINICAL_TRIAL = "clinical_trial"
    CONTRAINDICATION = "contraindication"
    MULTI_STEP = "multi_step"
    GENERAL = "general"


@dataclass
class QueryPlan:
    """Query execution plan."""
    query: str
    query_type: QueryType
    steps: List[Dict[str, Any]]
    context: Dict[str, Any]


@dataclass
class PlanningResponse:
    """Planning agent response."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    execution_plan: QueryPlan
    sub_responses: List[Union[DrugResponse, TrialResponse]]


class PlanningAgent:
    """Main orchestrating agent for complex pharmaceutical queries."""
    
    def __init__(self, llm_model: str = "gpt-4", temperature: float = 0.1):
        """Initialize the planning agent.
        
        Args:
            llm_model: Language model to use.
            temperature: LLM temperature setting.
        """
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Initialize specialized agents
        self.drug_agent = DrugAgent(llm_model, temperature)
        self.trial_agent = TrialAgent(llm_model, temperature)
        
        # Initialize prompts
        self._setup_prompts()
        
        logger.info("Initialized PlanningAgent")
    
    def _setup_prompts(self) -> None:
        """Setup prompt templates for query planning and synthesis."""
        
        self.planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a pharmaceutical research planning expert. Analyze the user's query and create an execution plan.

Your task is to:
1. Classify the query type
2. Identify required information sources
3. Break down complex queries into steps
4. Determine the optimal execution order

Query Types:
- drug_info: Basic drug information
- drug_interaction: Drug-drug interactions
- clinical_trial: Clinical trial search
- contraindication: Drug contraindications
- multi_step: Complex queries requiring multiple steps
- general: General pharmaceutical questions

Return your analysis in this format:
Query Type: [type]
Steps: [numbered list of steps]
Priority: [high/medium/low]
Complexity: [simple/moderate/complex]"""),
            ("human", "{query}")
        ])
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a pharmaceutical expert synthesizing information from multiple sources.

Guidelines:
- Integrate information from all sources coherently
- Highlight key findings and recommendations
- Note any conflicts or limitations in the data
- Provide clear, actionable conclusions
- Cite all sources appropriately
- Structure the response logically

Original Query: {original_query}
Drug Information: {drug_info}
Clinical Trial Information: {trial_info}
Additional Context: {context}"""),
            ("human", "Please synthesize this information into a comprehensive response.")
        ])
    
    def query(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> PlanningResponse:
        """Process a user query with intelligent planning and execution.
        
        Args:
            user_query: User's pharmaceutical query.
            context: Additional context information.
            
        Returns:
            PlanningResponse with comprehensive answer.
        """
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Create execution plan
        plan = self._create_plan(user_query, context or {})
        
        # Execute plan
        sub_responses = self._execute_plan(plan)
        
        # Synthesize final response
        final_response = self._synthesize_response(plan, sub_responses)
        
        return final_response
    
    def _create_plan(self, query: str, context: Dict[str, Any]) -> QueryPlan:
        """Create an execution plan for the query.
        
        Args:
            query: User query.
            context: Additional context.
            
        Returns:
            QueryPlan object.
        """
        try:
            # Use LLM to analyze query and create plan
            response = self.planning_prompt.format_prompt(query=query)
            messages = response.to_messages()
            ai_response = self.llm(messages)
            
            # Parse the response to extract plan details
            plan_text = ai_response.content.lower()
            
            # Determine query type
            query_type = QueryType.GENERAL
            if "drug_info" in plan_text:
                query_type = QueryType.DRUG_INFO
            elif "drug_interaction" in plan_text or "interaction" in plan_text:
                query_type = QueryType.DRUG_INTERACTION
            elif "clinical_trial" in plan_text or "trial" in plan_text:
                query_type = QueryType.CLINICAL_TRIAL
            elif "contraindication" in plan_text:
                query_type = QueryType.CONTRAINDICATION
            elif "multi_step" in plan_text or "complex" in plan_text:
                query_type = QueryType.MULTI_STEP
            
            # Create execution steps based on query type
            steps = self._generate_steps(query, query_type, context)
            
            return QueryPlan(
                query=query,
                query_type=query_type,
                steps=steps,
                context=context
            )
            
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            # Fallback to simple plan
            return QueryPlan(
                query=query,
                query_type=QueryType.GENERAL,
                steps=[{"agent": "drug", "action": "search", "query": query}],
                context=context
            )
    
    def _generate_steps(self, query: str, query_type: QueryType, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate execution steps based on query type.
        
        Args:
            query: User query.
            query_type: Classified query type.
            context: Additional context.
            
        Returns:
            List of execution steps.
        """
        steps = []
        
        if query_type == QueryType.DRUG_INFO:
            steps.append({
                "agent": "drug",
                "action": "get_info",
                "query": query,
                "priority": "high"
            })
        
        elif query_type == QueryType.DRUG_INTERACTION:
            steps.append({
                "agent": "drug",
                "action": "check_interaction",
                "query": query,
                "priority": "high"
            })
        
        elif query_type == QueryType.CLINICAL_TRIAL:
            steps.append({
                "agent": "trial",
                "action": "search",
                "query": query,
                "priority": "high"
            })
        
        elif query_type == QueryType.CONTRAINDICATION:
            steps.extend([
                {
                    "agent": "drug",
                    "action": "get_contraindications",
                    "query": query,
                    "priority": "high"
                },
                {
                    "agent": "trial",
                    "action": "search_safety",
                    "query": query,
                    "priority": "medium"
                }
            ])
        
        elif query_type == QueryType.MULTI_STEP:
            # For complex queries, use both agents
            steps.extend([
                {
                    "agent": "drug",
                    "action": "comprehensive_search",
                    "query": query,
                    "priority": "high"
                },
                {
                    "agent": "trial",
                    "action": "search",
                    "query": query,
                    "priority": "high"
                }
            ])
        
        else:  # GENERAL
            steps.append({
                "agent": "drug",
                "action": "search",
                "query": query,
                "priority": "medium"
            })
        
        return steps
    
    def _execute_plan(self, plan: QueryPlan) -> List[Union[DrugResponse, TrialResponse]]:
        """Execute the query plan.
        
        Args:
            plan: QueryPlan to execute.
            
        Returns:
            List of responses from specialized agents.
        """
        responses = []
        
        for step in plan.steps:
            try:
                agent_type = step["agent"]
                action = step["action"]
                query = step["query"]
                
                if agent_type == "drug":
                    response = self.drug_agent.process_query(query, plan.context)
                    responses.append(response)
                    
                elif agent_type == "trial":
                    response = self.trial_agent.process_query(query, plan.context)
                    responses.append(response)
                
                logger.debug(f"Executed step: {agent_type} - {action}")
                
            except Exception as e:
                logger.error(f"Step execution failed: {e}")
                continue
        
        return responses
    
    def _synthesize_response(self, plan: QueryPlan, sub_responses: List[Union[DrugResponse, TrialResponse]]) -> PlanningResponse:
        """Synthesize final response from sub-responses.
        
        Args:
            plan: Original query plan.
            sub_responses: Responses from specialized agents.
            
        Returns:
            Final PlanningResponse.
        """
        try:
            # Prepare information for synthesis
            drug_info = ""
            trial_info = ""
            all_sources = []
            total_confidence = 0.0
            
            for response in sub_responses:
                if isinstance(response, DrugResponse):
                    drug_info += f"\n{response.answer}"
                    all_sources.extend(response.sources)
                    total_confidence += response.confidence
                    
                elif isinstance(response, TrialResponse):
                    trial_info += f"\n{response.answer}"
                    all_sources.extend(response.sources)
                    total_confidence += response.confidence
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(sub_responses) if sub_responses else 0.0
            
            # Synthesize final response
            synthesis_response = self.synthesis_prompt.format_prompt(
                original_query=plan.query,
                drug_info=drug_info or "No drug information available.",
                trial_info=trial_info or "No clinical trial information available.",
                context=str(plan.context)
            )
            
            messages = synthesis_response.to_messages()
            ai_response = self.llm(messages)
            
            return PlanningResponse(
                answer=ai_response.content,
                sources=all_sources,
                confidence=avg_confidence,
                execution_plan=plan,
                sub_responses=sub_responses
            )
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            
            # Fallback response
            fallback_answer = "I apologize, but I encountered an error synthesizing the response. "
            if sub_responses:
                fallback_answer += "Here are the individual findings:\n\n"
                for i, response in enumerate(sub_responses, 1):
                    fallback_answer += f"{i}. {response.answer}\n\n"
            
            return PlanningResponse(
                answer=fallback_answer,
                sources=all_sources if 'all_sources' in locals() else [],
                confidence=0.5,
                execution_plan=plan,
                sub_responses=sub_responses
            )
