# PharmaGPT Development Guide

## Architecture Overview

PharmaGPT is built using a modular architecture with the following key components:

### Core Components

1. **Vector Store** (`vector_store/`)
   - `embedder.py`: Text embedding utilities using sentence transformers
   - `db_utils.py`: FAISS-based vector database operations

2. **Retrievers** (`retrievers/`)
   - `pubmed_retriever.py`: PubMed API integration for literature search
   - `drugbank_retriever.py`: DrugBank data retrieval (mock implementation)
   - `reranker.py`: Document reranking using cross-encoders

3. **Agents** (`agents/`)
   - `drug_agent.py`: Specialized agent for drug information and interactions
   - `trial_agent.py`: Clinical trial search and analysis agent
   - `planning_agent.py`: Main orchestrating agent using LangGraph-style planning

4. **Frontend** (`app/`)
   - `main.py`: Streamlit web interface

## Development Setup

### Prerequisites

- Python 3.8+
- OpenAI API key (for LLM functionality)
- Optional: Anthropic API key (for Claude models)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd PharmaGPT

# Install dependencies
make install-dev

# Setup environment
make setup

# Edit .env file with your API keys
nano .env
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
python -m pytest tests/test_pharmagpt.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all checks
make check
```

## Testing Strategy

### Unit Tests

- **Embedder Tests**: Test text preprocessing and embedding generation
- **Vector Store Tests**: Test document storage and retrieval
- **Retriever Tests**: Test data retrieval from various sources
- **Agent Tests**: Test agent initialization and basic functionality

### Integration Tests

- **End-to-End Tests**: Test complete query processing pipeline
- **API Integration Tests**: Test external API integrations (mocked)

### Sample Queries

The `data/samples/sample_queries.json` file contains test queries of varying complexity:

- **Simple**: Basic drug information queries
- **Moderate**: Drug interactions and clinical trial searches
- **Complex**: Multi-step reasoning and synthesis queries

## API Integration

### PubMed Integration

The PubMed retriever uses NCBI E-utilities:

- **Rate Limiting**: 3 requests/second with API key, 1/second without
- **Search**: Uses esearch.fcgi for finding PMIDs
- **Fetch**: Uses efetch.fcgi for retrieving article details
- **Parsing**: XML parsing for extracting article metadata

### DrugBank Integration

Currently uses mock data for demonstration:

- **Production**: Would require DrugBank license and API access
- **Mock Data**: Includes sample drugs (metformin, semaglutide)
- **Features**: Drug search, interaction checking, contraindication analysis

### Clinical Trials Integration

Mock implementation of ClinicalTrials.gov API:

- **Production**: Would use actual ClinicalTrials.gov API
- **Features**: Trial search, filtering by phase/status/age
- **Data**: Includes sample trials for demonstration

## Agent Architecture

### Planning Agent

The main orchestrating agent that:

1. **Analyzes** user queries to determine type and complexity
2. **Plans** execution steps using multiple specialized agents
3. **Coordinates** information retrieval from various sources
4. **Synthesizes** final responses from multiple sub-responses

### Drug Agent

Specialized for pharmaceutical information:

- **Drug Information**: Basic drug properties and mechanisms
- **Interactions**: Drug-drug interaction analysis
- **Contraindications**: Safety analysis for specific conditions
- **Literature Integration**: Combines DrugBank data with PubMed literature

### Trial Agent

Focused on clinical trial information:

- **Trial Search**: Finding relevant clinical trials
- **Analysis**: Interpreting trial design and outcomes
- **Filtering**: Age, condition, and intervention-based filtering
- **Literature Support**: Supporting evidence from published studies

## Advanced RAG Features

### Multi-Vector Search

- **Parallel Retrieval**: Simultaneous search across multiple sources
- **Source Fusion**: Combining results from different databases
- **Relevance Scoring**: Weighted scoring based on source reliability

### Reranking

- **Cross-Encoder**: Semantic relevance scoring
- **Hybrid Scoring**: Combines multiple signals:
  - Semantic similarity (60%)
  - Original retrieval score (20%)
  - Source reliability (10%)
  - Publication recency (10%)

### Chain-of-Thought

- **Query Decomposition**: Breaking complex queries into steps
- **Step-by-Step Reasoning**: Guided reasoning through prompts
- **Evidence Integration**: Combining evidence from multiple steps

## Deployment Considerations

### Environment Variables

Required configuration in `.env`:

```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here  # Optional
PUBMED_API_KEY=your_key_here     # Optional, for higher rate limits
```

### Performance Optimization

- **Caching**: Vector embeddings and API responses
- **Batch Processing**: Efficient handling of multiple documents
- **Rate Limiting**: Respectful API usage

### Scaling

- **Vector Store**: Can be replaced with cloud solutions (Pinecone, Weaviate)
- **LLM**: Supports multiple providers (OpenAI, Anthropic)
- **Deployment**: Streamlit app can be containerized

## Security & Compliance

### Data Privacy

- **No Storage**: User queries are not permanently stored
- **API Keys**: Securely managed through environment variables
- **Logging**: Configurable logging levels

### Medical Disclaimer

- **Educational Use**: System is for research and educational purposes
- **Professional Consultation**: Always recommend consulting healthcare professionals
- **Liability**: Clear disclaimers about medical advice limitations

## Extending the System

### Adding New Retrievers

1. Create new retriever class in `retrievers/`
2. Implement standard interface methods
3. Add to agent initialization
4. Update configuration

### Adding New Agents

1. Create agent class in `agents/`
2. Implement query processing methods
3. Add to planning agent orchestration
4. Create specialized prompts

### Custom Embeddings

1. Extend `TextEmbedder` class
2. Implement domain-specific preprocessing
3. Add medical/pharmaceutical vocabulary
4. Fine-tune on domain data

## Monitoring & Analytics

### Logging

- **Query Tracking**: Log user queries and responses
- **Performance Metrics**: Response times and confidence scores
- **Error Monitoring**: Track and analyze failures

### Metrics

- **Usage Statistics**: Query types and frequency
- **Quality Metrics**: User feedback and confidence scores
- **Performance**: Response times and resource usage

## Contributing

### Code Style

- **Formatting**: Black with 100-character line length
- **Imports**: isort with black profile
- **Type Hints**: Required for all public functions
- **Docstrings**: Google-style docstrings

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all checks pass
5. Submit pull request with description

### Issue Reporting

- **Bug Reports**: Include reproduction steps and environment details
- **Feature Requests**: Describe use case and expected behavior
- **Documentation**: Improvements and clarifications welcome
