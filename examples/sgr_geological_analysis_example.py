"""
Example: Using SGR Deep Research with Geological Schemas

This example demonstrates how to use SGR schemas for structured
geological analysis with the geomas pipeline.
"""

import os
import sys
from typing import Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import SGR Deep Research (if installed)
try:
    from sgr_deep_research import BaseAgent
    SGR_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    print(f"Note: SGR Deep Research not fully configured: {e}")
    print("Running in demonstration mode with schema validation only.")
    SGR_AVAILABLE = False
    
    # Mock BaseAgent for demonstration
    class BaseAgent:
        def __init__(self, llm_config, schema):
            self.llm_config = llm_config
            self.schema = schema
        
        def analyze(self, query, context_data):
            return {
                "query": query,
                "schema_type": self.schema.get("type"),
                "status": "simulated"
            }

# Import geomas schemas
from geomas.core.sgr_schemas import (
    RESOURCE_ASSESSMENT_SCHEMA,
    RISK_ANALYSIS_SCHEMA,
    ECONOMIC_VIABILITY_SCHEMA
)


class GeologicalAnalysisAgent:
    """
    SGR-powered geological analysis agent that uses structured schemas
    for systematic mineral deposit assessment.
    """
    
    def __init__(self, llm_config: Dict[str, Any] = None):
        """
        Initialize the geological analysis agent.
        
        Args:
            llm_config: Configuration for the LLM (model, API key, etc.)
        """
        if llm_config is None:
            llm_config = {"model": "demo", "api_key": "demo"}
        
        self.llm_config = llm_config
        self.schemas = {
            "resource_assessment": RESOURCE_ASSESSMENT_SCHEMA,
            "risk_analysis": RISK_ANALYSIS_SCHEMA,
            "economic_viability": ECONOMIC_VIABILITY_SCHEMA
        }
    
    def assess_resources(self, deposit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform structured resource assessment following JORC/NI 43-101 standards.
        
        Args:
            deposit_data: Dictionary containing drill holes, assays, geological maps
            
        Returns:
            Structured resource estimate with confidence levels
        """
        print("Initiating Resource Assessment...")
        print(f"Schema: {RESOURCE_ASSESSMENT_SCHEMA['type']}")
        print(f"Stages: {len(RESOURCE_ASSESSMENT_SCHEMA['stages'])}")
        
        # Initialize agent with resource assessment schema
        agent = BaseAgent(
            llm_config=self.llm_config,
            schema=RESOURCE_ASSESSMENT_SCHEMA
        )
        
        # Execute schema-guided analysis
        result = agent.analyze(
            query="Perform comprehensive resource assessment",
            context_data=deposit_data
        )
        
        return result
    
    def analyze_risks(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis across all risk categories.
        
        Args:
            project_data: Project information including geology, technical, ESG data
            
        Returns:
            Integrated risk assessment with mitigation strategies
        """
        print("\nInitiating Risk Analysis...")
        print(f"Schema: {RISK_ANALYSIS_SCHEMA['type']}")
        print(f"Stages: {len(RISK_ANALYSIS_SCHEMA['stages'])}")
        
        agent = BaseAgent(
            llm_config=self.llm_config,
            schema=RISK_ANALYSIS_SCHEMA
        )
        
        result = agent.analyze(
            query="Perform comprehensive risk assessment",
            context_data=project_data
        )
        
        return result
    
    def evaluate_economics(self, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate economic viability and calculate financial metrics.
        
        Args:
            economic_data: Resource estimate, costs, prices, and assumptions
            
        Returns:
            Financial analysis with NPV, IRR, sensitivity results
        """
        print("\nInitiating Economic Viability Analysis...")
        print(f"Schema: {ECONOMIC_VIABILITY_SCHEMA['type']}")
        print(f"Stages: {len(ECONOMIC_VIABILITY_SCHEMA['stages'])}")
        
        agent = BaseAgent(
            llm_config=self.llm_config,
            schema=ECONOMIC_VIABILITY_SCHEMA
        )
        
        result = agent.analyze(
            query="Evaluate economic viability and investment potential",
            context_data=economic_data
        )
        
        return result


def example_resource_assessment():
    """
    Example: Resource Assessment for a gold deposit.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: RESOURCE ASSESSMENT")
    print("="*70)
    
    # Simulated deposit data
    deposit_data = {
        "deposit_name": "Aginskoye Gold Deposit",
        "location": "Kamchatka, Russia",
        "drill_hole_data": {
            "total_holes": 45,
            "total_meters": 8500,
            "average_spacing": "50m x 50m"
        },
        "assay_results": {
            "gold_samples": 856,
            "average_grade": 43.7,  # g/t
            "grade_range": [0.1, 285.0]
        },
        "geological_setting": "Epithermal gold-silver vein system",
        "mineralization_style": "Quartz-sulfide veins"
    }
    
    # Display schema stages
    print("\nSchema Stages:")
    for i, stage_name in enumerate(RESOURCE_ASSESSMENT_SCHEMA['stages'].keys(), 1):
        stage = RESOURCE_ASSESSMENT_SCHEMA['stages'][stage_name]
        print(f"  {i}. {stage_name}")
        print(f"     => {stage['description']}")
    
    print("\nRequired Report Sections:")
    for section in RESOURCE_ASSESSMENT_SCHEMA['reporting']['required_sections']:
        print(f"  - {section}")
    
    # Note: Actual SGR execution would happen here
    print("\n[Simulated] Resource Assessment Complete")
    print("Output would include:")
    print("  - Measured Resources: X tonnes @ Y g/t Au")
    print("  - Indicated Resources: X tonnes @ Y g/t Au")
    print("  - Inferred Resources: X tonnes @ Y g/t Au")
    print("  - Confidence Level: High/Moderate/Low")


def example_risk_analysis():
    """
    Example: Risk Analysis for a mining project.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: RISK ANALYSIS")
    print("="*70)
    
    project_data = {
        "project_name": "Gold Mine Development",
        "stage": "Pre-feasibility",
        "resource_confidence": "Indicated + Inferred",
        "mining_method": "Underground",
        "environmental_sensitivities": ["Protected watershed", "Wildlife habitat"],
        "infrastructure": "Remote location, no existing access"
    }
    
    print("\nRisk Categories:")
    for stage_name in RISK_ANALYSIS_SCHEMA['stages'].keys():
        stage = RISK_ANALYSIS_SCHEMA['stages'][stage_name]
        print(f"  - {stage_name}")
        print(f"    {stage['description']}")
    
    print("\n[Simulated] Risk Analysis Complete")
    print("Output would include:")
    print("  - Overall Project Risk Rating: Moderate")
    print("  - Top 5 Critical Risks with Mitigation Plans")
    print("  - Residual Risk Level: Acceptable")


def example_economic_viability():
    """
    Example: Economic Viability Assessment.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: ECONOMIC VIABILITY ASSESSMENT")
    print("="*70)
    
    economic_data = {
        "resource_estimate": {
            "tonnage": 1_200_000,  # tonnes
            "grade": 3.5,  # g/t Au
            "metal_content": 135_000  # oz Au
        },
        "commodity_price": 2000,  # USD/oz
        "operating_cost": 85,  # USD/tonne
        "capex": 45_000_000,  # USD
        "mine_life": 8  # years
    }
    
    print("\nEconomic Analysis Stages:")
    for stage_name in ECONOMIC_VIABILITY_SCHEMA['stages'].keys():
        stage = ECONOMIC_VIABILITY_SCHEMA['stages'][stage_name]
        print(f"  - {stage_name}")
    
    print("\n[Simulated] Economic Analysis Complete")
    print("Output would include:")
    print("  - NPV (8%): $XX.X million")
    print("  - IRR: XX.X%")
    print("  - Payback Period: X.X years")
    print("  - Investment Recommendation: Proceed/Hold/Divest")


def main():
    """
    Run all examples.
    """
    print("\n" + "#"*70)
    print("# SGR GEOLOGICAL ANALYSIS EXAMPLES")
    print("# Structured reasoning for mineral deposit assessment")
    print("#"*70)
    
    # Display available schemas
    print("\nAvailable Schemas:")
    print(f"  1. Resource Assessment ({len(RESOURCE_ASSESSMENT_SCHEMA['stages'])} stages)")
    print(f"  2. Risk Analysis ({len(RISK_ANALYSIS_SCHEMA['stages'])} stages)")
    print(f"  3. Economic Viability ({len(ECONOMIC_VIABILITY_SCHEMA['stages'])} stages)")
    
    # Run examples
    example_resource_assessment()
    example_risk_analysis()
    example_economic_viability()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Configure LLM connection (OpenAI, local model, etc.)")
    print("  2. Integrate with geomas BERT NER for entity extraction")
    print("  3. Connect to vector database for context retrieval")
    print("  4. Deploy as API service or interactive interface")
    print("\nFor full integration, see: geomas/core/sgr_schemas/README.md")


if __name__ == "__main__":
    main()

