"""Demo script for PharmaGPT functionality."""

import os
import sys
import json
import logging
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.planning_agent import PlanningAgent
from agents.drug_agent import DrugAgent
from agents.trial_agent import TrialAgent
from retrievers.drugbank_retriever import DrugBankRetriever
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sample_queries():
    """Load sample queries from JSON file."""
    try:
        with open('data/samples/sample_queries.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Sample queries file not found. Using default queries.")
        return {
            "sample_queries": [
                {
                    "id": 1,
                    "query": "What are the side effects of metformin?",
                    "type": "drug_info",
                    "complexity": "simple"
                },
                {
                    "id": 2,
                    "query": "Find trials testing semaglutide in diabetes",
                    "type": "clinical_trial",
                    "complexity": "moderate"
                }
            ]
        }


def demo_drugbank_retriever():
    """Demonstrate DrugBank retriever functionality."""
    print("\n" + "="*60)
    print("DEMO: DrugBank Retriever")
    print("="*60)
    
    retriever = DrugBankRetriever()
    
    # Test drug search
    print("\n1. Searching for 'metformin':")
    drugs = retriever.search_drug("metformin")
    for drug in drugs:
        print(f"   - {drug.name} ({drug.drugbank_id})")
        print(f"     Indication: {drug.indication}")
        print(f"     Mechanism: {drug.mechanism_of_action[:100]}...")
    
    # Test drug interaction
    print("\n2. Checking interaction between metformin and alcohol:")
    interaction = retriever.check_drug_interaction("metformin", "alcohol")
    if interaction:
        print(f"   WARNING: Interaction found: {interaction}")
    else:
        print("   OK: No interaction found")

    # Test contraindications
    print("\n3. Checking contraindications for metformin in kidney disease:")
    contraindication = retriever.get_contraindications("metformin", "kidney disease")
    if contraindication:
        print(f"   WARNING: Contraindication: {contraindication}")
    else:
        print("   OK: No specific contraindication found")


def demo_drug_agent():
    """Demonstrate Drug Agent functionality."""
    print("\n" + "="*60)
    print("DEMO: Drug Agent")
    print("="*60)

    if not config.OPENAI_API_KEY:
        print("WARNING: OpenAI API key not configured. Skipping Drug Agent demo.")
        print("   Set OPENAI_API_KEY environment variable to test this feature.")
        return
    
    try:
        agent = DrugAgent()
        
        # Test basic drug query
        print("\n1. Processing query: 'What are the side effects of metformin?'")
        response = agent.process_query("What are the side effects of metformin?")
        
        print(f"   Answer: {response.answer[:200]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Sources: {len(response.sources)} sources found")
        
    except Exception as e:
        print(f"   ERROR: {e}")


def demo_trial_agent():
    """Demonstrate Trial Agent functionality."""
    print("\n" + "="*60)
    print("DEMO: Trial Agent")
    print("="*60)

    if not config.OPENAI_API_KEY:
        print("WARNING: OpenAI API key not configured. Skipping Trial Agent demo.")
        return
    
    try:
        agent = TrialAgent()
        
        # Test trial search
        print("\n1. Processing query: 'Find trials testing metformin in diabetes'")
        response = agent.process_query("Find trials testing metformin in diabetes")
        
        print(f"   Answer: {response.answer[:200]}...")
        print(f"   Trials found: {len(response.trials)}")
        print(f"   Confidence: {response.confidence:.2f}")
        
        # Show trial details
        for trial in response.trials[:2]:  # Show first 2 trials
            print(f"\n   Trial: {trial.title}")
            print(f"      NCT ID: {trial.nct_id}")
            print(f"      Status: {trial.status}")
            print(f"      Phase: {trial.phase}")

    except Exception as e:
        print(f"   ERROR: {e}")


def demo_planning_agent():
    """Demonstrate Planning Agent functionality."""
    print("\n" + "="*60)
    print("DEMO: Planning Agent (Main System)")
    print("="*60)

    if not config.OPENAI_API_KEY:
        print("WARNING: OpenAI API key not configured. Skipping Planning Agent demo.")
        return
    
    try:
        agent = PlanningAgent()
        
        # Load sample queries
        sample_data = load_sample_queries()
        
        # Test with a complex query
        complex_query = "What are the contraindications of combining metformin with semaglutide in diabetic patients?"
        print(f"\n1. Processing complex query:")
        print(f"   '{complex_query}'")
        
        context = {
            "patient_age": 65,
            "patient_conditions": ["type 2 diabetes", "hypertension"]
        }
        
        response = agent.query(complex_query, context)

        print(f"\n   Answer: {response.answer[:300]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Sources: {len(response.sources)} sources")
        print(f"   Sub-responses: {len(response.sub_responses)} agent responses")
        print(f"   Query type: {response.execution_plan.query_type.value}")

        # Show execution steps
        print(f"\n   Execution Plan:")
        for i, step in enumerate(response.execution_plan.steps, 1):
            print(f"      {i}. {step['agent'].title()} Agent - {step['action']}")

    except Exception as e:
        print(f"   ERROR: {e}")


def demo_sample_queries():
    """Demonstrate with sample queries."""
    print("\n" + "="*60)
    print("DEMO: Sample Queries")
    print("="*60)
    
    sample_data = load_sample_queries()
    
    print(f"\nLoaded {len(sample_data['sample_queries'])} sample queries:")
    
    for query_data in sample_data['sample_queries'][:3]:  # Show first 3
        print(f"\n{query_data['id']}. {query_data['query']}")
        print(f"   Type: {query_data['type']}")
        print(f"   Complexity: {query_data['complexity']}")
        print(f"   Expected sources: {', '.join(query_data['expected_sources'])}")


def main():
    """Run the complete demo."""
    print("PharmaGPT Demo")
    print("=" * 60)
    print("This demo showcases the capabilities of PharmaGPT,")
    print("an intelligent drug & clinical trial assistant using advanced RAG.")

    # Check configuration
    print(f"\nConfiguration:")
    print(f"   OpenAI API Key: {'Configured' if config.OPENAI_API_KEY else 'Not configured'}")
    print(f"   Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"   Reranker Model: {config.RERANKER_MODEL}")
    
    # Run demos
    try:
        demo_drugbank_retriever()
        demo_sample_queries()
        demo_drug_agent()
        demo_trial_agent()
        demo_planning_agent()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    print("\nTo run the full application:")
    print("   streamlit run app/main.py")
    print("\nTo run tests:")
    print("   python -m pytest tests/test_pharmagpt.py -v")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
