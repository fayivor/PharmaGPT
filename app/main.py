"""Streamlit frontend for PharmaGPT."""

import streamlit as st
import logging
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planning_agent import PlanningAgent, PlanningResponse
from agents.drug_agent import DrugResponse
from agents.trial_agent import TrialResponse
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PharmaGPT - Intelligent Drug & Clinical Trial Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_agent():
    """Initialize the PharmaGPT planning agent."""
    try:
        return PlanningAgent()
    except Exception as e:
        st.error(f"Failed to initialize PharmaGPT: {e}")
        return None


def display_confidence(confidence: float) -> str:
    """Display confidence score with appropriate styling."""
    if confidence >= 0.8:
        return f'<span class="confidence-high">High ({confidence:.1%})</span>'
    elif confidence >= 0.6:
        return f'<span class="confidence-medium">Medium ({confidence:.1%})</span>'
    else:
        return f'<span class="confidence-low">Low ({confidence:.1%})</span>'


def display_sources(sources: list) -> None:
    """Display sources in an organized manner."""
    if not sources:
        return
    
    st.subheader("Sources")
    
    # Group sources by type
    drug_sources = [s for s in sources if s.get('type') == 'drugbank']
    pubmed_sources = [s for s in sources if s.get('type') == 'pubmed']
    trial_sources = [s for s in sources if s.get('type') == 'clinical_trial']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if drug_sources:
            st.write("**DrugBank**")
            for source in drug_sources:
                st.markdown(f"""
                <div class="source-box">
                    <strong>{source.get('name', 'Unknown Drug')}</strong><br>
                    ID: {source.get('id', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if pubmed_sources:
            st.write("**PubMed Literature**")
            for source in pubmed_sources:
                st.markdown(f"""
                <div class="source-box">
                    <strong>{source.get('title', 'Unknown Title')[:50]}...</strong><br>
                    PMID: {source.get('pmid', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        if trial_sources:
            st.write("**Clinical Trials**")
            for source in trial_sources:
                st.markdown(f"""
                <div class="source-box">
                    <strong>{source.get('title', 'Unknown Trial')[:50]}...</strong><br>
                    NCT ID: {source.get('nct_id', 'N/A')}<br>
                    Status: {source.get('status', 'N/A')}
                </div>
                """, unsafe_allow_html=True)


def display_detailed_results(response: PlanningResponse) -> None:
    """Display detailed results from sub-agents."""
    if not response.sub_responses:
        return
    
    st.subheader("Detailed Analysis")
    
    for i, sub_response in enumerate(response.sub_responses):
        with st.expander(f"Analysis {i+1}: {type(sub_response).__name__}"):
            st.write(sub_response.answer)
            
            if isinstance(sub_response, DrugResponse) and sub_response.drug_info:
                st.write("**Drug Information:**")
                for drug in sub_response.drug_info:
                    st.write(f"- **{drug.name}** ({drug.drugbank_id})")
                    st.write(f"  - Indication: {drug.indication}")
                    st.write(f"  - Mechanism: {drug.mechanism_of_action}")
            
            elif isinstance(sub_response, TrialResponse) and sub_response.trials:
                st.write("**Clinical Trials:**")
                for trial in sub_response.trials:
                    st.write(f"- **{trial.title}**")
                    st.write(f"  - NCT ID: {trial.nct_id}")
                    st.write(f"  - Status: {trial.status}")
                    st.write(f"  - Phase: {trial.phase}")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">PharmaGPT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An Intelligent Drug & Clinical Trial Assistant Using Advanced RAG</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password", 
                               value=config.OPENAI_API_KEY or "",
                               help="Enter your OpenAI API key")
        
        if api_key:
            config.OPENAI_API_KEY = api_key
        
        st.header("Query Examples")
        example_queries = [
            "What are the contraindications of combining metformin with semaglutide in diabetic patients?",
            "Find clinical trials testing metformin with GLP-1 inhibitors in patients over 60",
            "What are the side effects of semaglutide?",
            "Summarize the latest research on cardiovascular benefits of GLP-1 agonists"
        ]
        
        for i, example in enumerate(example_queries):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.query_input = example
    
    # Initialize agent
    agent = initialize_agent()
    if not agent:
        st.error("Failed to initialize PharmaGPT. Please check your configuration.")
        return
    
    # Main query interface
    st.header("Ask PharmaGPT")
    
    # Query input
    query = st.text_area(
        "Enter your pharmaceutical or clinical trial question:",
        height=100,
        value=st.session_state.get('query_input', ''),
        placeholder="e.g., What are the drug interactions between metformin and insulin?"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_age = st.number_input("Patient Age (optional)", min_value=0, max_value=120, value=0)
            patient_conditions = st.text_input("Patient Conditions (optional)", 
                                             placeholder="e.g., diabetes, hypertension")
        
        with col2:
            query_type = st.selectbox("Query Type", 
                                    ["Auto-detect", "Drug Information", "Drug Interactions", 
                                     "Clinical Trials", "Contraindications"])
            include_trials = st.checkbox("Include Clinical Trials", value=True)
    
    # Submit button
    if st.button("Ask PharmaGPT", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
            return
        
        if not config.OPENAI_API_KEY:
            st.error("Please provide an OpenAI API key in the sidebar.")
            return
        
        # Prepare context
        context = {}
        if patient_age > 0:
            context['patient_age'] = patient_age
        if patient_conditions:
            context['patient_conditions'] = patient_conditions
        if query_type != "Auto-detect":
            context['preferred_type'] = query_type.lower().replace(" ", "_")
        context['include_trials'] = include_trials
        
        # Process query
        with st.spinner("Analyzing your question and searching for information..."):
            try:
                response = agent.query(query, context)
                
                # Display main response
                st.header("Answer")
                st.write(response.answer)
                
                # Display confidence
                st.markdown(f"**Confidence:** {display_confidence(response.confidence)}", 
                           unsafe_allow_html=True)
                
                # Display sources
                display_sources(response.sources)
                
                # Display detailed results
                display_detailed_results(response)
                
                # Display execution plan
                with st.expander("Execution Plan"):
                    st.write(f"**Query Type:** {response.execution_plan.query_type.value}")
                    st.write("**Steps Executed:**")
                    for i, step in enumerate(response.execution_plan.steps, 1):
                        st.write(f"{i}. {step['agent'].title()} Agent - {step['action']}")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Query processing failed: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Disclaimer:</strong> This tool is for educational and research purposes only.
        Always consult healthcare professionals for medical advice.</p>
        <p>Built using Streamlit, LangChain, and advanced RAG techniques.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
