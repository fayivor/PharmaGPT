"""Clinical trial search and analysis agent for PharmaGPT."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from retrievers.pubmed_retriever import PubMedRetriever
from retrievers.reranker import HybridReranker
from config import config

logger = logging.getLogger(__name__)


@dataclass
class ClinicalTrial:
    """Clinical trial representation."""
    nct_id: str
    title: str
    brief_summary: str
    detailed_description: str
    status: str
    phase: str
    study_type: str
    conditions: List[str]
    interventions: List[str]
    primary_outcomes: List[str]
    secondary_outcomes: List[str]
    enrollment: int
    start_date: str
    completion_date: str
    sponsor: str
    locations: List[str]
    eligibility_criteria: str
    age_range: str


@dataclass
class TrialQuery:
    """Clinical trial query representation."""
    query: str
    conditions: List[str]
    interventions: List[str]
    phase: Optional[str]
    status: Optional[str]
    age_criteria: Optional[str]
    other_criteria: Dict[str, Any]


@dataclass
class TrialResponse:
    """Trial agent response."""
    answer: str
    trials: List[ClinicalTrial]
    sources: List[Dict[str, Any]]
    confidence: float


class ClinicalTrialsRetriever:
    """Retriever for ClinicalTrials.gov data."""
    
    def __init__(self):
        """Initialize the clinical trials retriever."""
        self.base_url = config.CLINICALTRIALS_BASE_URL
        self.session = requests.Session()
        logger.info("Initialized ClinicalTrialsRetriever")
    
    def search_trials(self, query: str, max_results: int = 20, 
                     filters: Optional[Dict[str, Any]] = None) -> List[ClinicalTrial]:
        """Search for clinical trials.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            filters: Additional search filters.
            
        Returns:
            List of ClinicalTrial objects.
        """
        # This is a simplified mock implementation
        # In practice, you would use the actual ClinicalTrials.gov API
        
        mock_trials = self._get_mock_trials()
        
        # Simple filtering based on query
        query_lower = query.lower()
        filtered_trials = []
        
        for trial in mock_trials:
            if (query_lower in trial.title.lower() or 
                query_lower in trial.brief_summary.lower() or
                any(query_lower in condition.lower() for condition in trial.conditions) or
                any(query_lower in intervention.lower() for intervention in trial.interventions)):
                filtered_trials.append(trial)
        
        # Apply additional filters
        if filters:
            filtered_trials = self._apply_filters(filtered_trials, filters)
        
        logger.info(f"Found {len(filtered_trials)} trials for query: {query[:50]}...")
        return filtered_trials[:max_results]
    
    def _get_mock_trials(self) -> List[ClinicalTrial]:
        """Get mock trial data for demonstration."""
        return [
            ClinicalTrial(
                nct_id="NCT12345678",
                title="Metformin and Semaglutide Combination in Type 2 Diabetes",
                brief_summary="A randomized controlled trial comparing metformin plus semaglutide versus metformin alone in patients with type 2 diabetes.",
                detailed_description="This study evaluates the efficacy and safety of combining semaglutide with metformin in adults with type 2 diabetes mellitus who have inadequate glycemic control on metformin monotherapy.",
                status="Recruiting",
                phase="Phase 3",
                study_type="Interventional",
                conditions=["Type 2 Diabetes Mellitus"],
                interventions=["Metformin", "Semaglutide"],
                primary_outcomes=["Change in HbA1c from baseline"],
                secondary_outcomes=["Weight loss", "Time to cardiovascular events"],
                enrollment=500,
                start_date="2023-01-15",
                completion_date="2025-12-31",
                sponsor="Pharmaceutical Research Institute",
                locations=["United States", "Canada", "Europe"],
                eligibility_criteria="Adults 18-75 years with T2DM, HbA1c 7-10%, on stable metformin dose",
                age_range="18-75 years"
            ),
            ClinicalTrial(
                nct_id="NCT87654321",
                title="Cardiovascular Outcomes with GLP-1 Agonists in Elderly Patients",
                brief_summary="Long-term cardiovascular safety study of GLP-1 receptor agonists in patients over 65 years with diabetes.",
                detailed_description="A prospective observational study examining cardiovascular outcomes in elderly patients with type 2 diabetes treated with GLP-1 receptor agonists.",
                status="Active, not recruiting",
                phase="Phase 4",
                study_type="Observational",
                conditions=["Type 2 Diabetes Mellitus", "Cardiovascular Disease"],
                interventions=["GLP-1 Receptor Agonists"],
                primary_outcomes=["Major adverse cardiovascular events"],
                secondary_outcomes=["All-cause mortality", "Hospitalization rates"],
                enrollment=2000,
                start_date="2020-06-01",
                completion_date="2024-06-01",
                sponsor="Cardiovascular Research Foundation",
                locations=["United States", "United Kingdom", "Germany"],
                eligibility_criteria="Adults ≥65 years with T2DM and established CVD or high CV risk",
                age_range="≥65 years"
            )
        ]
    
    def _apply_filters(self, trials: List[ClinicalTrial], filters: Dict[str, Any]) -> List[ClinicalTrial]:
        """Apply filters to trial list.
        
        Args:
            trials: List of trials to filter.
            filters: Filter criteria.
            
        Returns:
            Filtered list of trials.
        """
        filtered = trials
        
        if 'phase' in filters:
            phase = filters['phase'].lower()
            filtered = [t for t in filtered if phase in t.phase.lower()]
        
        if 'status' in filters:
            status = filters['status'].lower()
            filtered = [t for t in filtered if status in t.status.lower()]
        
        if 'age_min' in filters:
            age_min = filters['age_min']
            # Simplified age filtering
            filtered = [t for t in filtered if 'age' in t.eligibility_criteria.lower()]
        
        return filtered


class TrialAgent:
    """Agent specialized in clinical trial search and analysis."""
    
    def __init__(self, llm_model: str = "gpt-4", temperature: float = 0.1):
        """Initialize the trial agent.
        
        Args:
            llm_model: Language model to use.
            temperature: LLM temperature setting.
        """
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.trials_retriever = ClinicalTrialsRetriever()
        self.pubmed_retriever = PubMedRetriever()
        self.reranker = HybridReranker()
        
        # Initialize prompts
        self._setup_prompts()
        
        logger.info("Initialized TrialAgent")
    
    def _setup_prompts(self) -> None:
        """Setup prompt templates for trial analysis."""
        
        self.trial_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a clinical research expert specializing in clinical trial analysis and interpretation.

Guidelines:
- Provide comprehensive analysis of clinical trials
- Explain study design, endpoints, and significance
- Discuss limitations and potential biases
- Compare multiple trials when relevant
- Cite specific trial identifiers and sources
- Consider patient populations and generalizability

Clinical Trials: {trials}
Supporting Literature: {literature}"""),
            ("human", "{query}")
        ])
        
        self.trial_search_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a clinical research expert helping users find relevant clinical trials.

Guidelines:
- Summarize key trial characteristics
- Explain eligibility criteria clearly
- Highlight primary and secondary endpoints
- Discuss trial status and timeline
- Provide contact information when available
- Consider patient-specific factors

Search Results: {trials}
Query Context: {context}"""),
            ("human", "{query}")
        ])

    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> TrialResponse:
        """Process a clinical trial query.

        Args:
            query: User query about clinical trials.
            context: Additional context.

        Returns:
            TrialResponse with answer and trial information.
        """
        logger.info(f"Processing trial query: {query[:100]}...")

        # Parse query to extract search criteria
        parsed_query = self._parse_query(query, context or {})

        # Search for relevant trials
        trials = self._search_trials(parsed_query)

        # Retrieve supporting literature
        literature = self._retrieve_literature(query, parsed_query)

        # Generate response
        response = self._generate_response(parsed_query, trials, literature)

        return response

    def _parse_query(self, query: str, context: Dict[str, Any]) -> TrialQuery:
        """Parse query to extract trial search criteria.

        Args:
            query: User query.
            context: Additional context.

        Returns:
            Parsed TrialQuery object.
        """
        query_lower = query.lower()

        # Extract conditions
        conditions = []
        condition_keywords = ["diabetes", "cancer", "hypertension", "cardiovascular", "obesity"]
        for keyword in condition_keywords:
            if keyword in query_lower:
                conditions.append(keyword)

        # Extract interventions/drugs
        interventions = []
        drug_keywords = ["metformin", "semaglutide", "insulin", "glp-1", "sglt2"]
        for keyword in drug_keywords:
            if keyword in query_lower:
                interventions.append(keyword)

        # Extract phase
        phase = None
        if "phase 1" in query_lower or "phase i" in query_lower:
            phase = "Phase 1"
        elif "phase 2" in query_lower or "phase ii" in query_lower:
            phase = "Phase 2"
        elif "phase 3" in query_lower or "phase iii" in query_lower:
            phase = "Phase 3"
        elif "phase 4" in query_lower or "phase iv" in query_lower:
            phase = "Phase 4"

        # Extract status
        status = None
        if "recruiting" in query_lower:
            status = "Recruiting"
        elif "completed" in query_lower:
            status = "Completed"
        elif "active" in query_lower:
            status = "Active"

        # Extract age criteria
        age_criteria = None
        if "elderly" in query_lower or "over 65" in query_lower or ">65" in query_lower:
            age_criteria = "elderly"
        elif "pediatric" in query_lower or "children" in query_lower:
            age_criteria = "pediatric"

        return TrialQuery(
            query=query,
            conditions=conditions,
            interventions=interventions,
            phase=phase,
            status=status,
            age_criteria=age_criteria,
            other_criteria=context
        )

    def _search_trials(self, parsed_query: TrialQuery) -> List[ClinicalTrial]:
        """Search for clinical trials based on parsed query.

        Args:
            parsed_query: Parsed query object.

        Returns:
            List of relevant clinical trials.
        """
        # Build search filters
        filters = {}
        if parsed_query.phase:
            filters['phase'] = parsed_query.phase
        if parsed_query.status:
            filters['status'] = parsed_query.status
        if parsed_query.age_criteria == "elderly":
            filters['age_min'] = 65

        # Search trials
        trials = self.trials_retriever.search_trials(
            parsed_query.query,
            max_results=10,
            filters=filters
        )

        logger.debug(f"Found {len(trials)} trials")
        return trials

    def _retrieve_literature(self, query: str, parsed_query: TrialQuery) -> List[Dict[str, Any]]:
        """Retrieve supporting literature for the trial query.

        Args:
            query: Original query.
            parsed_query: Parsed query object.

        Returns:
            List of literature sources.
        """
        # Construct PubMed search query
        search_terms = []
        search_terms.extend(parsed_query.conditions)
        search_terms.extend(parsed_query.interventions)
        search_terms.append("clinical trial")

        pubmed_query = " AND ".join([f'"{term}"' for term in search_terms])

        try:
            articles = self.pubmed_retriever.search_and_fetch(
                pubmed_query,
                max_results=5,
                filters={"article_type": "Clinical Trial"}
            )

            # Convert to format for reranking
            documents = []
            for article in articles:
                content = f"{article.title}. {article.abstract}"
                metadata = {
                    "pmid": article.pmid,
                    "title": article.title,
                    "journal": article.journal,
                    "publication_date": article.publication_date,
                    "source": "pubmed"
                }
                documents.append((content, 1.0, metadata))

            # Rerank documents
            if documents:
                ranked_docs = self.reranker.rerank(query, documents, top_k=3)
                return [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": doc.rerank_score
                    }
                    for doc in ranked_docs
                ]

        except Exception as e:
            logger.error(f"Literature retrieval failed: {e}")

        return []

    def _generate_response(self, parsed_query: TrialQuery, trials: List[ClinicalTrial],
                          literature: List[Dict[str, Any]]) -> TrialResponse:
        """Generate response based on query and retrieved information.

        Args:
            parsed_query: Parsed query object.
            trials: Retrieved clinical trials.
            literature: Retrieved literature.

        Returns:
            TrialResponse object.
        """
        try:
            # Format trials and literature for prompt
            trials_text = self._format_trials(trials)
            literature_text = self._format_literature(literature)

            # Select appropriate prompt
            if len(trials) > 1:
                prompt = self.trial_analysis_prompt
                response = prompt.format_prompt(
                    query=parsed_query.query,
                    trials=trials_text,
                    literature=literature_text
                )
            else:
                prompt = self.trial_search_prompt
                response = prompt.format_prompt(
                    query=parsed_query.query,
                    trials=trials_text,
                    context=str(parsed_query.other_criteria)
                )

            # Generate response
            messages = response.to_messages()
            ai_response = self.llm(messages)

            # Calculate confidence
            confidence = self._calculate_confidence(trials, literature)

            # Prepare sources
            sources = []
            for trial in trials:
                sources.append({
                    "type": "clinical_trial",
                    "nct_id": trial.nct_id,
                    "title": trial.title,
                    "status": trial.status
                })

            for lit in literature:
                sources.append({
                    "type": "pubmed",
                    "pmid": lit["metadata"].get("pmid"),
                    "title": lit["metadata"].get("title")
                })

            return TrialResponse(
                answer=ai_response.content,
                trials=trials,
                sources=sources,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return TrialResponse(
                answer="I apologize, but I encountered an error processing your trial query. Please try again.",
                trials=trials,
                sources=[],
                confidence=0.0
            )

    def _format_trials(self, trials: List[ClinicalTrial]) -> str:
        """Format trials for prompt.

        Args:
            trials: List of ClinicalTrial objects.

        Returns:
            Formatted trials text.
        """
        if not trials:
            return "No relevant clinical trials found."

        formatted = []
        for trial in trials:
            trial_text = f"""
Trial: {trial.title}
NCT ID: {trial.nct_id}
Status: {trial.status}
Phase: {trial.phase}
Study Type: {trial.study_type}
Conditions: {', '.join(trial.conditions)}
Interventions: {', '.join(trial.interventions)}
Primary Outcomes: {', '.join(trial.primary_outcomes)}
Secondary Outcomes: {', '.join(trial.secondary_outcomes)}
Enrollment: {trial.enrollment}
Age Range: {trial.age_range}
Eligibility: {trial.eligibility_criteria}
Sponsor: {trial.sponsor}
Locations: {', '.join(trial.locations)}
Start Date: {trial.start_date}
Completion Date: {trial.completion_date}
Summary: {trial.brief_summary}
"""
            formatted.append(trial_text)

        return "\n".join(formatted)

    def _format_literature(self, literature: List[Dict[str, Any]]) -> str:
        """Format literature for prompt.

        Args:
            literature: List of literature sources.

        Returns:
            Formatted literature text.
        """
        if not literature:
            return "No relevant literature found."

        formatted = []
        for i, lit in enumerate(literature, 1):
            metadata = lit["metadata"]
            lit_text = f"""
[{i}] {metadata.get('title', 'Unknown Title')}
Journal: {metadata.get('journal', 'Unknown')}
Date: {metadata.get('publication_date', 'Unknown')}
PMID: {metadata.get('pmid', 'Unknown')}
Content: {lit['content'][:400]}...
"""
            formatted.append(lit_text)

        return "\n".join(formatted)

    def _calculate_confidence(self, trials: List[ClinicalTrial], literature: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the response.

        Args:
            trials: Retrieved trials.
            literature: Retrieved literature.

        Returns:
            Confidence score between 0 and 1.
        """
        confidence = 0.3  # Base confidence

        # Increase confidence based on available information
        if trials:
            confidence += 0.4 * min(len(trials) / 3, 1.0)

        if literature:
            confidence += 0.3 * min(len(literature) / 3, 1.0)

        return min(confidence, 1.0)
