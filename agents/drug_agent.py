"""Drug information agent for PharmaGPT."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from retrievers.drugbank_retriever import DrugBankRetriever, DrugInfo
from retrievers.pubmed_retriever import PubMedRetriever
from retrievers.reranker import HybridReranker
from config import config

logger = logging.getLogger(__name__)


@dataclass
class DrugQuery:
    """Drug-related query representation."""
    query: str
    drug_names: List[str]
    query_type: str  # 'interaction', 'side_effects', 'mechanism', 'contraindication'
    context: Dict[str, Any]


@dataclass
class DrugResponse:
    """Drug agent response."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    drug_info: List[DrugInfo]


class DrugAgent:
    """Agent specialized in drug information and interactions."""
    
    def __init__(self, llm_model: str = "gpt-4", temperature: float = 0.1):
        """Initialize the drug agent.
        
        Args:
            llm_model: Language model to use.
            temperature: LLM temperature setting.
        """
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.drugbank_retriever = DrugBankRetriever()
        self.pubmed_retriever = PubMedRetriever()
        self.reranker = HybridReranker()
        
        # Initialize prompts
        self._setup_prompts()
        
        logger.info("Initialized DrugAgent")
    
    def _setup_prompts(self) -> None:
        """Setup prompt templates for different query types."""
        
        self.drug_info_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a pharmaceutical expert assistant. Provide accurate, evidence-based information about drugs.

Guidelines:
- Always cite your sources
- Mention any limitations or uncertainties
- Include relevant warnings or contraindications
- Use clear, professional language
- Structure your response logically

Context: {context}
Drug Information: {drug_info}
Literature: {literature}"""),
            ("human", "{query}")
        ])
        
        self.interaction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a pharmaceutical expert specializing in drug interactions. Analyze potential interactions between medications.

Guidelines:
- Assess interaction severity (major, moderate, minor)
- Explain the mechanism of interaction
- Provide clinical recommendations
- Cite evidence sources
- Include monitoring recommendations

Drug 1 Information: {drug1_info}
Drug 2 Information: {drug2_info}
Literature Evidence: {literature}"""),
            ("human", "{query}")
        ])
        
        self.contraindication_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a pharmaceutical expert analyzing drug contraindications and safety profiles.

Guidelines:
- Identify absolute and relative contraindications
- Explain the clinical rationale
- Consider patient-specific factors
- Provide alternative recommendations if appropriate
- Cite evidence sources

Drug Information: {drug_info}
Patient Context: {patient_context}
Literature: {literature}"""),
            ("human", "{query}")
        ])
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> DrugResponse:
        """Process a drug-related query.
        
        Args:
            query: User query about drugs.
            context: Additional context (patient info, etc.).
            
        Returns:
            DrugResponse with answer and sources.
        """
        logger.info(f"Processing drug query: {query[:100]}...")
        
        # Parse query to extract drug names and determine type
        parsed_query = self._parse_query(query, context or {})
        
        # Retrieve drug information
        drug_info = self._retrieve_drug_info(parsed_query.drug_names)
        
        # Retrieve literature evidence
        literature = self._retrieve_literature(query, parsed_query.drug_names)
        
        # Generate response based on query type
        response = self._generate_response(parsed_query, drug_info, literature)
        
        return response
    
    def _parse_query(self, query: str, context: Dict[str, Any]) -> DrugQuery:
        """Parse query to extract drug names and determine query type.
        
        Args:
            query: User query.
            context: Additional context.
            
        Returns:
            Parsed DrugQuery object.
        """
        query_lower = query.lower()
        
        # Simple drug name extraction (in practice, use NER)
        potential_drugs = []
        common_drugs = ["metformin", "semaglutide", "insulin", "aspirin", "warfarin", 
                       "lisinopril", "atorvastatin", "amlodipine", "losartan"]
        
        for drug in common_drugs:
            if drug in query_lower:
                potential_drugs.append(drug)
        
        # Determine query type
        query_type = "general"
        if any(word in query_lower for word in ["interact", "combination", "together"]):
            query_type = "interaction"
        elif any(word in query_lower for word in ["side effect", "adverse", "toxicity"]):
            query_type = "side_effects"
        elif any(word in query_lower for word in ["mechanism", "how does", "works"]):
            query_type = "mechanism"
        elif any(word in query_lower for word in ["contraindication", "avoid", "should not"]):
            query_type = "contraindication"
        
        return DrugQuery(
            query=query,
            drug_names=potential_drugs,
            query_type=query_type,
            context=context
        )
    
    def _retrieve_drug_info(self, drug_names: List[str]) -> List[DrugInfo]:
        """Retrieve drug information from DrugBank.
        
        Args:
            drug_names: List of drug names to look up.
            
        Returns:
            List of DrugInfo objects.
        """
        drug_info = []
        
        for drug_name in drug_names:
            drugs = self.drugbank_retriever.search_drug(drug_name)
            drug_info.extend(drugs)
        
        logger.debug(f"Retrieved information for {len(drug_info)} drugs")
        return drug_info
    
    def _retrieve_literature(self, query: str, drug_names: List[str]) -> List[Dict[str, Any]]:
        """Retrieve relevant literature from PubMed.
        
        Args:
            query: Original query.
            drug_names: List of drug names.
            
        Returns:
            List of literature sources.
        """
        # Construct PubMed search query
        if drug_names:
            pubmed_query = " AND ".join([f'"{drug}"' for drug in drug_names])
            if len(drug_names) == 1:
                pubmed_query += " AND (pharmacology OR mechanism OR interaction OR safety)"
        else:
            pubmed_query = query
        
        try:
            articles = self.pubmed_retriever.search_and_fetch(
                pubmed_query, 
                max_results=10,
                filters={"article_type": "Review"}
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
                ranked_docs = self.reranker.rerank(query, documents, top_k=5)
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
    
    def _generate_response(self, parsed_query: DrugQuery, drug_info: List[DrugInfo], 
                          literature: List[Dict[str, Any]]) -> DrugResponse:
        """Generate response based on query type and retrieved information.
        
        Args:
            parsed_query: Parsed query object.
            drug_info: Retrieved drug information.
            literature: Retrieved literature.
            
        Returns:
            DrugResponse object.
        """
        try:
            # Prepare context
            drug_info_text = self._format_drug_info(drug_info)
            literature_text = self._format_literature(literature)
            
            # Select appropriate prompt based on query type
            if parsed_query.query_type == "interaction" and len(drug_info) >= 2:
                prompt = self.interaction_prompt
                response = prompt.format_prompt(
                    query=parsed_query.query,
                    drug1_info=drug_info_text,
                    drug2_info="",  # Simplified for this example
                    literature=literature_text
                )
            elif parsed_query.query_type == "contraindication":
                prompt = self.contraindication_prompt
                response = prompt.format_prompt(
                    query=parsed_query.query,
                    drug_info=drug_info_text,
                    patient_context=str(parsed_query.context),
                    literature=literature_text
                )
            else:
                prompt = self.drug_info_prompt
                response = prompt.format_prompt(
                    query=parsed_query.query,
                    context=str(parsed_query.context),
                    drug_info=drug_info_text,
                    literature=literature_text
                )
            
            # Generate response
            messages = response.to_messages()
            ai_response = self.llm(messages)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(drug_info, literature)
            
            # Prepare sources
            sources = []
            for drug in drug_info:
                sources.append({
                    "type": "drugbank",
                    "id": drug.drugbank_id,
                    "name": drug.name
                })
            
            for lit in literature:
                sources.append({
                    "type": "pubmed",
                    "pmid": lit["metadata"].get("pmid"),
                    "title": lit["metadata"].get("title")
                })
            
            return DrugResponse(
                answer=ai_response.content,
                sources=sources,
                confidence=confidence,
                drug_info=drug_info
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return DrugResponse(
                answer="I apologize, but I encountered an error processing your query. Please try again.",
                sources=[],
                confidence=0.0,
                drug_info=drug_info
            )
    
    def _format_drug_info(self, drug_info: List[DrugInfo]) -> str:
        """Format drug information for prompt.
        
        Args:
            drug_info: List of DrugInfo objects.
            
        Returns:
            Formatted drug information text.
        """
        if not drug_info:
            return "No specific drug information available."
        
        formatted = []
        for drug in drug_info:
            drug_text = f"""
Drug: {drug.name} (DrugBank ID: {drug.drugbank_id})
Description: {drug.description}
Indication: {drug.indication}
Mechanism of Action: {drug.mechanism_of_action}
Pharmacodynamics: {drug.pharmacodynamics}
Toxicity: {drug.toxicity}
Drug Interactions: {'; '.join([f"{i['drug']}: {i['description']}" for i in drug.drug_interactions])}
Categories: {', '.join(drug.categories)}
"""
            formatted.append(drug_text)
        
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
Content: {lit['content'][:500]}...
"""
            formatted.append(lit_text)
        
        return "\n".join(formatted)
    
    def _calculate_confidence(self, drug_info: List[DrugInfo], literature: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the response.
        
        Args:
            drug_info: Retrieved drug information.
            literature: Retrieved literature.
            
        Returns:
            Confidence score between 0 and 1.
        """
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available information
        if drug_info:
            confidence += 0.3
        
        if literature:
            confidence += 0.2 * min(len(literature) / 3, 1.0)
        
        return min(confidence, 1.0)
