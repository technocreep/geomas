# SGR Schemas for Geological Analysis

This directory contains **Schema-Guided Reasoning (SGR)** schemas designed for structured geological analysis using the SGR Deep Research framework.

## Overview

SGR schemas define the reasoning process, data requirements, validation rules, and output structures for different types of geological analyses. These schemas guide the AI agent through systematic evaluation and ensure comprehensive, standardized assessments.

## Available Schemas

### 1. Resource Assessment Schema (`resource_assessment.py`)

**Purpose**: Comprehensive mineral resource estimation following international standards (JORC, NI 43-101, SAMREC)

**Key Stages**:
- Data Collection and Validation
- Geological Interpretation and Domain Definition
- Grade Estimation and Resource Classification
- Uncertainty Analysis and Risk Quantification

**Output**: Technical report with resource estimate statement, confidence levels, and recommendations

**Use Cases**:
- Initial resource estimation
- Resource update and reconciliation
- Due diligence assessments
- Feasibility study support

---

### 2. Risk Analysis Schema (`risk_analysis.py`)

**Purpose**: Comprehensive risk assessment for exploration and development projects

**Key Stages**:
- Geological Risk Assessment (resource confidence, grade variability)
- Technical Risk Assessment (mining, processing, infrastructure)
- Environmental and Social Risk (ESG compliance, permitting)
- Market and Economic Risk (commodity prices, costs)
- Integrated Risk Matrix (prioritized mitigation strategies)

**Output**: Risk assessment report with prioritized risks and mitigation plans

**Use Cases**:
- Project risk evaluation
- Investment decision support
- Operational planning
- Stakeholder communication

---

### 3. Economic Viability Schema (`economic_viability.py`)

**Purpose**: Economic assessment and investment decision framework

**Key Stages**:
- Resource Valuation (in-situ value calculation)
- Operating Cost Estimation (mining, processing, G&A)
- Capital Cost Estimation (initial and sustaining)
- Financial Modeling (NPV, IRR, payback)
- Sensitivity Analysis (key value drivers)
- Investment Decision Framework

**Output**: Economic assessment report with financial metrics and investment recommendation

**Use Cases**:
- Scoping and feasibility studies
- Investment decisions
- Strategic planning
- Portfolio optimization

---

## Schema Structure

Each schema follows a consistent structure:

```python
{
    "type": "workflow_type",
    "version": "1.0",
    "description": "Schema purpose",
    
    "stages": {
        "stage_name": {
            "description": "What this stage does",
            "input_data": ["required", "data", "sources"],
            "reasoning_steps": ["step1", "step2", "..."],
            "output_structure": {
                # Structured output definition
            },
            "validation_rules": ["rule1", "rule2"]
        }
    },
    
    "reporting": {
        "template": "report_type",
        "required_sections": ["section1", "section2"],
        "citation_requirements": {...}
    },
    
    "quality_checks": {
        "mandatory_validations": [...],
        "warning_triggers": [...]
    },
    
    "entity_mapping": {
        # Maps to geomas BERT NER entity types
    }
}
```

## Integration with Geomas

These schemas integrate with the geomas pipeline:

```
PDF Documents → Chunks → BERT NER (17 entity types) → Vector DB
                                                          ↓
                                              SGR Agent (Schema-Guided Analysis)
                                                          ↓
                                              Structured Reports + Insights
```

### Entity Type Mapping

SGR schemas leverage geomas BERT NER entity extraction:

| Geomas Entity Type | Used in Schema |
|-------------------|----------------|
| RESOURCE_POTENTIAL | Resource Assessment, Economic Viability |
| ORE_COMPONENT | Resource Assessment, Economic Viability |
| ORE_BODIES | Resource Assessment |
| TECHNOLOGICAL | Risk Analysis, Economic Viability |
| STRUCTURAL_TECTONIC | Risk Analysis |
| GEO_CHEMICAL | Resource Assessment |
| STUDY_INFO | All schemas |
| INFO_SOURCES | All schemas (for citations) |

## Usage Example

```python
from sgr_deep_research import BaseAgent
from geomas.core.sgr_schemas import (
    RESOURCE_ASSESSMENT_SCHEMA,
    RISK_ANALYSIS_SCHEMA,
    ECONOMIC_VIABILITY_SCHEMA
)

# Initialize SGR Agent with schema
agent = BaseAgent(
    llm_config=llm_config,
    schema=RESOURCE_ASSESSMENT_SCHEMA
)

# Run analysis
result = agent.analyze(
    query="Assess the resource potential of Aginskoye gold deposit",
    context_data={
        "drill_holes": drill_data,
        "assay_results": assay_data,
        "geological_maps": maps
    }
)

# Access structured output
resource_estimate = result['stages']['grade_estimation']['resource_estimate']
print(f"Indicated Resources: {resource_estimate['indicated_resources']['tonnage']} tonnes")
print(f"Average Grade: {resource_estimate['indicated_resources']['grade']} g/t")
```

## Quality Assurance

All schemas include:

1. **Validation Rules**: Ensure data quality and methodology appropriateness
2. **Quality Checks**: Mandatory validations before proceeding
3. **Warning Triggers**: Flag potential issues for review
4. **Citation Requirements**: Ensure traceability and evidence-based analysis

## Extending Schemas

To create a new schema:

1. Copy an existing schema as template
2. Define stages with clear input/output structures
3. Specify validation rules for each stage
4. Document reporting requirements
5. Add to `__init__.py`
6. Update this README

## Standards Compliance

Schemas are designed to align with:

- **JORC Code 2012** (Australasia)
- **NI 43-101** (Canada)
- **SAMREC Code** (South Africa)
- **PERC Reporting Standard** (Europe)

## Future Enhancements

Planned schema additions:

- [ ] Exploration Target Definition
- [ ] Metallurgical Testing Analysis
- [ ] Environmental Impact Assessment
- [ ] Mine Planning Optimization
- [ ] Geotechnical Risk Assessment
- [ ] Hydrology and Water Management

## References

- SGR Deep Research: https://github.com/vamplabAI/sgr-deep-research
- JORC Code 2012: http://www.jorc.org
- NI 43-101 Standards: https://www.osc.ca

