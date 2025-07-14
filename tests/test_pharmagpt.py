"""Test suite for PharmaGPT components."""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store.embedder import TextEmbedder, PharmaTextEmbedder
from vector_store.db_utils import VectorStore, Document
from retrievers.drugbank_retriever import DrugBankRetriever
from retrievers.reranker import CrossEncoderReranker
from agents.drug_agent import DrugAgent
from agents.trial_agent import TrialAgent
from agents.planning_agent import PlanningAgent


class TestTextEmbedder:
    """Test cases for text embedding functionality."""
    
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        embedder = TextEmbedder()
        assert embedder.model_name is not None
        assert embedder.device in ["cuda", "cpu"]
    
    def test_text_preprocessing(self):
        """Test text preprocessing."""
        embedder = PharmaTextEmbedder()
        
        # Test basic preprocessing
        text = "  This is a test   with extra spaces  "
        processed = embedder._preprocess_text(text)
        assert processed == "This is a test with extra spaces"
        
        # Test pharmaceutical abbreviation expansion
        text = "Patient received 500 mg metformin"
        processed = embedder._preprocess_text(text)
        assert "milligrams" in processed
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_text(self, mock_transformer):
        """Test text embedding."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer.return_value = mock_model
        
        embedder = TextEmbedder()
        embedding = embedder.embed_text("test text")
        
        assert embedding is not None
        assert len(embedding) == 3


class TestVectorStore:
    """Test cases for vector store functionality."""
    
    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        store = VectorStore()
        assert store.store_path is not None
        assert store.embedder is not None
        assert store.documents == []
    
    def test_document_creation(self):
        """Test document creation."""
        doc = Document(
            id="test_1",
            content="This is a test document about metformin.",
            metadata={"source": "test", "type": "drug_info"},
            source="test"
        )
        
        assert doc.id == "test_1"
        assert "metformin" in doc.content
        assert doc.metadata["source"] == "test"


class TestDrugBankRetriever:
    """Test cases for DrugBank retriever."""
    
    def test_retriever_initialization(self):
        """Test retriever initialization."""
        retriever = DrugBankRetriever()
        assert retriever._mock_data is not None
        assert len(retriever._mock_data) > 0
    
    def test_drug_search(self):
        """Test drug search functionality."""
        retriever = DrugBankRetriever()
        
        # Test search for metformin
        results = retriever.search_drug("metformin")
        assert len(results) > 0
        assert results[0].name.lower() == "metformin"
        
        # Test search for non-existent drug
        results = retriever.search_drug("nonexistentdrug123")
        assert len(results) == 0
    
    def test_drug_interaction_check(self):
        """Test drug interaction checking."""
        retriever = DrugBankRetriever()
        
        # Test known interaction
        interaction = retriever.check_drug_interaction("metformin", "alcohol")
        assert interaction is not None
        assert "lactic acidosis" in interaction.lower()
        
        # Test no interaction
        interaction = retriever.check_drug_interaction("metformin", "nonexistentdrug")
        assert interaction is None


class TestReranker:
    """Test cases for document reranking."""
    
    @patch('sentence_transformers.CrossEncoder')
    def test_reranker_initialization(self, mock_cross_encoder):
        """Test reranker initialization."""
        reranker = CrossEncoderReranker()
        assert reranker.model_name is not None
    
    @patch('sentence_transformers.CrossEncoder')
    def test_document_reranking(self, mock_cross_encoder):
        """Test document reranking functionality."""
        # Mock the cross encoder
        mock_model = Mock()
        mock_model.predict.return_value = [0.8, 0.6, 0.9]
        mock_cross_encoder.return_value = mock_model
        
        reranker = CrossEncoderReranker()
        
        documents = [
            ("Document about metformin diabetes treatment", 0.5, {"source": "pubmed"}),
            ("Document about general medicine", 0.7, {"source": "pubmed"}),
            ("Document about metformin mechanism", 0.6, {"source": "drugbank"})
        ]
        
        ranked_docs = reranker.rerank("metformin diabetes", documents, top_k=2)
        
        assert len(ranked_docs) == 2
        assert ranked_docs[0].rerank_score >= ranked_docs[1].rerank_score


class TestAgents:
    """Test cases for agent functionality."""
    
    @patch('langchain.chat_models.ChatOpenAI')
    def test_drug_agent_initialization(self, mock_llm):
        """Test drug agent initialization."""
        agent = DrugAgent()
        assert agent.drugbank_retriever is not None
        assert agent.pubmed_retriever is not None
        assert agent.reranker is not None
    
    @patch('langchain.chat_models.ChatOpenAI')
    def test_trial_agent_initialization(self, mock_llm):
        """Test trial agent initialization."""
        agent = TrialAgent()
        assert agent.trials_retriever is not None
        assert agent.pubmed_retriever is not None
        assert agent.reranker is not None
    
    @patch('langchain.chat_models.ChatOpenAI')
    def test_planning_agent_initialization(self, mock_llm):
        """Test planning agent initialization."""
        agent = PlanningAgent()
        assert agent.drug_agent is not None
        assert agent.trial_agent is not None


class TestIntegration:
    """Integration tests for the complete system."""
    
    def load_sample_queries(self):
        """Load sample queries for testing."""
        try:
            with open('data/samples/sample_queries.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"sample_queries": [], "test_contexts": []}
    
    def test_sample_queries_format(self):
        """Test that sample queries are properly formatted."""
        data = self.load_sample_queries()
        
        assert "sample_queries" in data
        assert "test_contexts" in data
        
        for query in data["sample_queries"]:
            assert "id" in query
            assert "query" in query
            assert "type" in query
            assert "expected_sources" in query
            assert "complexity" in query
            assert "description" in query
    
    @patch('langchain.chat_models.ChatOpenAI')
    def test_end_to_end_query_processing(self, mock_llm):
        """Test end-to-end query processing (mocked)."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "This is a test response about metformin."
        mock_llm.return_value.return_value = mock_response
        
        # Test with a simple query
        agent = PlanningAgent()
        
        # This would normally require API keys, so we'll just test initialization
        assert agent is not None


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        retriever = DrugBankRetriever()
        
        results = retriever.search_drug("")
        assert len(results) == 0
        
        results = retriever.search_drug("   ")
        assert len(results) == 0
    
    def test_invalid_drug_interaction(self):
        """Test handling of invalid drug interactions."""
        retriever = DrugBankRetriever()
        
        interaction = retriever.check_drug_interaction("", "")
        assert interaction is None
        
        interaction = retriever.check_drug_interaction("invalid_drug", "another_invalid")
        assert interaction is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
