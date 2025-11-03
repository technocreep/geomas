"""
Example: SGR-Enhanced RAG API Integration for Geological Analysis.

This example demonstrates how to use the enhanced RagSGRApi with
SGR (Schema-Guided Reasoning) for structured geological analysis.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Example demonstrating the API usage
def main():
    print("=" * 80)
    print("SGR-ENHANCED RAG API INTEGRATION EXAMPLE")
    print("=" * 80)
    
    print("\n### Integration Overview ###\n")
    
    print("The enhanced RagSGRApi extends traditional RAG with:")
    print("  1. BERT NER for geological entity extraction (17 entity types)")
    print("  2. SGR Deep Research for structured reasoning")
    print("  3. Schema-guided analysis (resource, risk, economic)")
    print("  4. Domain-specific geological prompts")
    
    print("\n### Usage Patterns ###\n")
    
    # Pattern 1: Traditional RAG mode
    print("## Pattern 1: Traditional RAG (backward compatible)")
    print("""
from geomas.api import RagSGRApi
from langchain_community.llms import Ollama  # or any LLM

# Initialize without SGR
llm = Ollama(model="mistral")
rag_api = RagSGRApi(
    llm=llm,
    use_sgr=False,  # Traditional mode
    use_geological_prompts=True  # Use geological prompts
)

# Use like original RagApi
response = rag_api.eval(
    user_prompt="What are the gold grades at Aginskoye deposit?",
    retrievers=[retriever],
    collection_names=["geological_docs"],
    rerank=True
)
print(response)  # String response
    """)
    
    # Pattern 2: SGR mode with entity extraction
    print("\n## Pattern 2: SGR Mode with Entity Extraction")
    print("""
from geomas.api import RagSGRApi

# Initialize with SGR and NER
rag_api = RagSGRApi(
    llm=llm,
    use_sgr=True,  # Enable SGR
    bert_ner_model_path="./bert_ner_output",  # Path to BERT NER model
    use_geological_prompts=True
)

# Query with automatic analysis type detection
response = rag_api.eval(
    user_prompt="Assess the resource potential of Aginskoye deposit",
    retrievers=[retriever],
    collection_names=["geological_docs"],
    rerank=True,
    use_entities=True  # Extract entities
)

# Response is a structured dictionary
print("Analysis Type:", response["schema_type"])
print("Entities Found:", response["entity_summary"])
print("Status:", response["status"])
    """)
    
    # Pattern 3: Structured analysis with explicit schema
    print("\n## Pattern 3: Structured Analysis with Explicit Schema")
    print("""
# Explicit resource assessment
response = rag_api.eval_structured(
    user_prompt="Estimate resources for Aginskoye gold deposit",
    retrievers=[retriever],
    collection_names=["geological_docs"],
    analysis_type="resource_assessment",  # Explicit schema
    rerank=True
)

# Structured output following JORC/NI 43-101 format
print("Data Quality:", response["stages_completed"][0])  # data_collection
print("Resource Estimate:", response["stages_completed"][2])  # grade_estimation

# Risk analysis example
risk_response = rag_api.eval_structured(
    user_prompt="What are the main risks for this project?",
    retrievers=[retriever],
    collection_names=["geological_docs"],
    analysis_type="risk_analysis"
)

# Economic analysis example
econ_response = rag_api.eval_structured(
    user_prompt="Evaluate economic viability",
    retrievers=[retriever],
    collection_names=["geological_docs"],
    analysis_type="economic_viability"
)
    """)
    
    # Pattern 4: Full pipeline
    print("\n## Pattern 4: Complete Pipeline with All Components")
    print("""
from geomas.api import RagSGRApi
from geomas.core.rag_modules.steps.retriever import DocsSearcherModels, Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

# Initialize components
llm = Ollama(model="mistral")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create retriever
docs_searcher = DocsSearcherModels(
    embedding_model=embedding_model,
    chroma_client=chroma_client
)
retriever = Retriever(top_k=5, docs_searcher_models=docs_searcher)

# Initialize enhanced RAG API
rag_api = RagSGRApi(
    llm=llm,
    use_sgr=True,
    bert_ner_model_path="./bert_ner_output",
    use_geological_prompts=True
)

# Query geological database
query = "Provide a comprehensive resource assessment for the Aginskoye gold deposit"

result = rag_api.eval(
    user_prompt=query,
    retrievers=[retriever],
    collection_names=["aginskoye_docs"],
    rerank=True,
    use_entities=True
)

# Access structured results
print(f"Analysis Type: {result['schema_type']}")
print(f"Documents Analyzed: {result['context_summary']['documents_count']}")
print(f"Entities Extracted: {result['entities_extracted']['entity_count']}")
print(f"Entity Types Found: {', '.join(result['entities_extracted']['relevant_types'])}")

# Entity breakdown
for entity_type, entities in result['entities_extracted']['processed_entities'].items():
    print(f"\\n{entity_type}:")
    for entity in entities[:3]:  # Show top 3
        print(f"  - {entity['text']} (confidence: {entity['confidence']:.2f})")
    """)
    
    print("\n### Available Analysis Types ###\n")
    print("  - resource_assessment: JORC/NI 43-101 compliant resource estimation")
    print("  - risk_analysis: Geological, technical, ESG, economic risk assessment")
    print("  - economic_viability: Financial analysis (NPV, IRR, sensitivity)")
    print("  - general: Auto-detected or general geological queries")
    
    print("\n### Geological Entity Types (17) ###\n")
    entity_types = [
        "GENERAL_INFO", "ORE_COMPONENT", "RESOURCE_POTENTIAL",
        "ORE_FORMATION", "MINERALOGICAL", "TECHNOLOGICAL",
        "STRATIGRAPHY", "STRUCTURAL_TECTONIC", "ORE_BODIES",
        "ORE_COMPOSITION", "GEODYNAMIC", "GEO_CHEMICAL",
        "METALLOGENIC_CHAR", "METASOMATIC", "FORMATION_CONDITIONS",
        "STUDY_INFO", "INFO_SOURCES"
    ]
    for i, et in enumerate(entity_types, 1):
        print(f"  {i:2d}. {et}")
    
    print("\n### Migration from Original RagApi ###\n")
    print("""
# Old code (RagApi):
from geomas.api.rag import RagApi
rag = RagApi(llm=llm)
response = rag.eval(query, retrievers, collections)

# New code (RagSGRApi) - fully backward compatible:
from geomas.api import RagSGRApi
rag = RagSGRApi(llm=llm, use_sgr=False)  # Same behavior as RagApi
response = rag.eval(query, retrievers, collections)

# Or enable new features:
rag = RagSGRApi(llm=llm, use_sgr=True, bert_ner_model_path="./bert_ner_output")
response = rag.eval(query, retrievers, collections, use_entities=True)
    """)
    
    print("\n### Configuration Requirements ###\n")
    print("""
1. config.yaml must exist in project root (for full SGR support)
2. BERT NER model trained on geological entities (optional but recommended)
3. ChromaDB with geological documents indexed
4. LLM configured (OpenAI, Anthropic, Ollama, or local model)

Minimal config.yaml:
```yaml
llm_provider: "openai"
openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
  temperature: 0.3

geomas:
  bert_ner_model_path: "./bert_ner_output"
  vector_db_path: "./chroma_db"
  default_collection: "geological_documents"
```
    """)
    
    print("\n" + "=" * 80)
    print("Integration complete! RagSGRApi is ready for use.")
    print("=" * 80)
    
    print("\n### Next Steps ###\n")
    print("1. Train or load BERT NER model for entity extraction")
    print("2. Index geological documents in ChromaDB")
    print("3. Configure LLM provider in config.yaml")
    print("4. Test with sample queries")
    print("5. Deploy as API service (FastAPI integration available)")
    
    print("\nFor detailed documentation, see:")
    print("  - geomas/core/sgr_schemas/README.md")
    print("  - geomas/api/rag_sgr.py")
    print("  - examples/sgr_geological_analysis_example.py")


if __name__ == "__main__":
    main()

