# PharmaGPT – An Intelligent Drug & Clinical Trial Assistant Using Advanced RAG

## Overview

**PharmaGPT** is an agentic AI assistant that answers complex pharmaceutical and clinical trial questions by retrieving and reasoning over multiple sources such as DrugBank, PubMed, FDA documents, and ClinicalTrials.gov.

> PharmaGPT is an advanced RAG-powered agent that assists in pharmaceutical queries and clinical trial navigation. Due to NDA restrictions, this repo contains a simplified version that reflects the techniques and thought processes used in my real-world work. It showcases advanced RAG features like multi-source retrieval, reranking, agentic task breakdown, and citation-backed answers.

## Advanced RAG Techniques Used

| Feature                        | Description                                                                               |
| ------------------------------ | ----------------------------------------------------------------------------------------- |
| **Multi-vector search**        | One query hits multiple sources (DrugBank, PubMed, Trials).                               |
| **Reranking**                  | Retrieved chunks reranked using cross-encoder or ColBERT for semantic relevance.          |
| **Chain-of-Thought Prompting** | Guides the model to reason step-by-step.                                                  |
| **Source-aware citation**      | Returns answers with document citations.                                                  |
| **LangGraph-style agent flow** | Breaks multi-hop queries into subtasks (e.g., drug lookup → trial match → age filtering). |
| **Fallback strategy**          | If vector store fails, agent uses tool like Bing Search or PubMed API.                    |

## Use Case Scenarios

1. **"What are the contraindications of combining Drug A with Drug B in diabetic patients?"**
2. **"Summarize the latest FDA warning on semaglutide and its implications."**
3. **"Find a trial that tested metformin with GLP-1 inhibitors in patients over 60."**

## Technology Stack

- **LangGraph** for agentic orchestration
- **FAISS** for Vector DB
- **OpenAI GPT-4** or **Claude 3** for generation
- **PubMed / ClinicalTrials.gov / DrugBank APIs** for retrieval
- **Streamlit** frontend

## Project Structure

```
pharma-gpt/
│
├── agents/
│   ├── trial_agent.py      # Clinical trial search and analysis
│   ├── drug_agent.py       # Drug information and interactions
│   └── planning_agent.py   # Query planning and orchestration
│
├── retrievers/
│   ├── pubmed_retriever.py    # PubMed API integration
│   ├── drugbank_retriever.py  # DrugBank data retrieval
│   └── reranker.py           # Document reranking
│
├── vector_store/
│   ├── embedder.py         # Text embedding utilities
│   └── db_utils.py         # Vector database operations
│
├── app/
│   └── main.py             # Streamlit interface
│
├── data/
│   └── samples/            # Sample data and test cases
│
├── README.md
└── requirements.txt
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PharmaGPT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

```bash
streamlit run app/main.py
```

## Configuration

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PUBMED_API_KEY=your_pubmed_api_key
```

## Usage Examples

### Basic Query
```python
from agents.planning_agent import PlanningAgent

agent = PlanningAgent()
response = agent.query("What are the side effects of metformin?")
print(response)
```

### Complex Multi-step Query
```python
query = "Find clinical trials for diabetes drugs that showed cardiovascular benefits in patients over 65"
response = agent.query(query)
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

## Disclaimer

This tool is for educational and research purposes only. Always consult healthcare professionals for medical advice.
