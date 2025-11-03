"""
Geological Analysis Prompts for SGR-Guided Reasoning.

These prompts are designed for structured geological analysis
using SGR Deep Research framework.
"""

from langchain_core.prompts import PromptTemplate


# System prompt for geological expert
GEOLOGICAL_EXPERT_SYSTEM_PROMPT = """You are an expert geologist and mineral resource professional with extensive experience in:
- Mineral resource estimation and classification (JORC, NI 43-101, SAMREC standards)
- Geological interpretation and modeling
- Risk assessment for mining projects
- Economic evaluation of mineral deposits
- Technical report preparation

Your role is to provide accurate, evidence-based analysis using only the information provided in the context documents and extracted geological entities. If information is insufficient, clearly state what additional data would be needed.

Key principles:
1. Base all conclusions on provided evidence
2. Use appropriate geological terminology
3. Follow international reporting standards
4. Clearly distinguish between facts and interpretations
5. Quantify uncertainty where possible
6. Cite sources from context when making statements
"""


# Resource Assessment Prompt
GEOLOGICAL_RESOURCE_ASSESSMENT_TEMPLATE = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}

You are performing a RESOURCE ASSESSMENT analysis following JORC/NI 43-101 standards.

Focus on:
- Data quality and completeness
- Geological interpretation and mineralization domains
- Grade estimation methodology
- Resource classification (Measured, Indicated, Inferred)
- Uncertainty quantification

<|eot_id|>
<|start_header_id|>user<|end_header_id|>

QUERY: {question}

GEOLOGICAL ENTITIES EXTRACTED:
{entities_summary}

CONTEXT DOCUMENTS:
{context_paragraphs}

Provide a structured resource assessment addressing:
1. Data inventory and quality
2. Geological model interpretation
3. Resource estimate (tonnage, grade, classification)
4. Confidence levels and uncertainties
5. Recommendations for additional work

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


# Risk Analysis Prompt
GEOLOGICAL_RISK_ANALYSIS_TEMPLATE = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}

You are performing a RISK ANALYSIS for a mining/exploration project.

Focus on:
- Geological risks (resource confidence, grade variability)
- Technical risks (mining, processing, infrastructure)
- Environmental and social risks
- Economic risks (commodity prices, costs)
- Risk mitigation strategies

<|eot_id|>
<|start_header_id|>user<|end_header_id|>

QUERY: {question}

GEOLOGICAL ENTITIES EXTRACTED:
{entities_summary}

CONTEXT DOCUMENTS:
{context_paragraphs}

Provide a structured risk assessment addressing:
1. Geological Risk Assessment (resource confidence, continuity, grade variability)
2. Technical Risk Assessment (mining method, processing, infrastructure)
3. Environmental & Social Risks (compliance, permitting, stakeholders)
4. Economic Risks (prices, costs, market)
5. Integrated Risk Matrix with prioritized mitigation strategies

For each risk, specify:
- Severity (Critical/High/Medium/Low)
- Likelihood (Very Likely/Likely/Possible/Unlikely)
- Mitigation measures
- Residual risk level

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


# Economic Viability Prompt
GEOLOGICAL_ECONOMIC_VIABILITY_TEMPLATE = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}

You are performing an ECONOMIC VIABILITY ASSESSMENT for a mineral deposit.

Focus on:
- Resource valuation (in-situ value)
- Operating cost estimation
- Capital cost estimation
- Financial metrics (NPV, IRR, payback)
- Sensitivity analysis

<|eot_id|>
<|start_header_id|>user<|end_header_id|>

QUERY: {question}

GEOLOGICAL ENTITIES EXTRACTED:
{entities_summary}

CONTEXT DOCUMENTS:
{context_paragraphs}

Provide a structured economic assessment addressing:
1. Resource Valuation (gross in-situ value, value by category/commodity)
2. Operating Costs (mining, processing, G&A, unit costs)
3. Capital Costs (initial capex, sustaining capital, breakdown)
4. Financial Metrics (NPV at various discount rates, IRR, payback period)
5. Sensitivity Analysis (price, cost, grade sensitivity)
6. Investment Recommendation (proceed/hold/divest with rationale)

Include:
- Key assumptions and their justification
- Benchmark comparison with peer operations
- Upside/base case/downside scenarios
- Critical success factors

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


# General Geological Query Prompt
GEOLOGICAL_GENERAL_QUERY_TEMPLATE = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Вы - эксперт-геолог, специализирующийся на анализе минеральных месторождений.
Отвечайте на вопросы, используя ТОЛЬКО информацию из предоставленных документов и извлеченных геологических сущностей.

Правила:
1. Используйте только предоставленную информацию для ответа
2. Если информации недостаточно, четко укажите это
3. Используйте корректную геологическую терминологию
4. Количественные данные приводите с единицами измерения
5. Ссылайтесь на источники информации из контекста

<|eot_id|>
<|start_header_id|>user<|end_header_id|>

ВОПРОС: {question}

ИЗВЛЕЧЕННЫЕ ГЕОЛОГИЧЕСКИЕ СУЩНОСТИ:
{entities_summary}

КОНТЕКСТНЫЕ ДОКУМЕНТЫ:
{context_paragraphs}

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


# Entity Summary Formatter
ENTITY_SUMMARY_TEMPLATE = """
Extracted {entity_count} geological entities:

{entity_details}

Key Information:
{key_info}

Schema Relevance:
{schema_relevance}
"""


# Create PromptTemplate objects
PROMPT_GEOLOGICAL_RESOURCE_ASSESSMENT = PromptTemplate(
    template=GEOLOGICAL_RESOURCE_ASSESSMENT_TEMPLATE,
    input_variables=["system_prompt", "question", "entities_summary", "context_paragraphs"]
)

PROMPT_GEOLOGICAL_RISK_ANALYSIS = PromptTemplate(
    template=GEOLOGICAL_RISK_ANALYSIS_TEMPLATE,
    input_variables=["system_prompt", "question", "entities_summary", "context_paragraphs"]
)

PROMPT_GEOLOGICAL_ECONOMIC_VIABILITY = PromptTemplate(
    template=GEOLOGICAL_ECONOMIC_VIABILITY_TEMPLATE,
    input_variables=["system_prompt", "question", "entities_summary", "context_paragraphs"]
)

PROMPT_GEOLOGICAL_GENERAL_QUERY = PromptTemplate(
    template=GEOLOGICAL_GENERAL_QUERY_TEMPLATE,
    input_variables=["question", "entities_summary", "context_paragraphs"]
)


# Reranking prompt for geological documents
GEOLOGICAL_RERANK_TEMPLATE = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Вы - система оценки релевантности геологических документов.

Оцените в 10-балльной шкале, насколько ТЕКСТ содержит полезную информацию для ответа на ЗАПРОС.
Геологический контекст должен учитываться при оценке релевантности.

Правила оценки:
1. Сначала напишите рассуждение о соответствии текста запросу
2. Учитывайте специфику геологической терминологии
3. Высокая оценка - текст напрямую отвечает на вопрос
4. Средняя оценка - текст содержит косвенно связанную информацию
5. Низкая оценка - текст не относится к запросу

Формат вывода:
Рассуждение: [ваше объяснение]
ОЦЕНКА: [число от 1 до 10]

<|eot_id|>
<|start_header_id|>user<|end_header_id|>

ЗАПРОС:
{question}

ТЕКСТ:
{context}

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

PROMPT_GEOLOGICAL_RERANK = PromptTemplate(
    template=GEOLOGICAL_RERANK_TEMPLATE,
    input_variables=["question", "context"]
)


def format_entities_summary(processed_entities: dict) -> str:
    """
    Format processed entities into readable summary for prompt.
    
    Args:
        processed_entities: Output from EntityProcessor.process_entities()
        
    Returns:
        Formatted string for prompt injection
    """
    if not processed_entities or processed_entities.get("entity_count", 0) == 0:
        return "No geological entities were extracted from the documents."
    
    parts = []
    
    # Entity counts
    entity_count = processed_entities["entity_count"]
    relevant_types = processed_entities["relevant_types"]
    parts.append(f"Total entities extracted: {entity_count} across {len(relevant_types)} types")
    
    # List by type
    parts.append("\nEntities by type:")
    for entity_type in relevant_types[:10]:  # Limit to top 10 types
        entities = processed_entities["processed_entities"][entity_type]
        sample_texts = [e["text"] for e in entities[:3]]  # Show top 3 examples
        parts.append(f"  - {entity_type} ({len(entities)}): {', '.join(sample_texts)}")
    
    # Key information
    key_info = processed_entities.get("key_information", {})
    if key_info:
        parts.append("\nKey Information:")
        if "ore_components" in key_info:
            parts.append(f"  Ore Components: {', '.join(key_info['ore_components'])}")
        if "resources" in key_info:
            parts.append(f"  Resource Info: {', '.join(key_info['resources'])}")
    
    return "\n".join(parts)


def format_context_paragraphs(documents: list, max_length: int = 3000) -> str:
    """
    Format context documents into numbered paragraphs for prompt.
    
    Args:
        documents: List of Document objects
        max_length: Maximum total character length
        
    Returns:
        Formatted string with numbered paragraphs
    """
    if not documents:
        return "No context documents available."
    
    paragraphs = []
    current_length = 0
    
    for i, doc in enumerate(documents, 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
        
        para = f"Параграф {i} (Источник: {source}):\n{content}\n"
        
        if current_length + len(para) > max_length:
            break
        
        paragraphs.append(para)
        current_length += len(para)
    
    return "\n".join(paragraphs)

