"""
Schema Validator for SGR Schemas.

Utilities to validate schema structure and completeness.
"""

from typing import Dict, List, Any, Tuple


class SchemaValidator:
    """
    Validates SGR schema structure and ensures all required components are present.
    """
    
    REQUIRED_TOP_LEVEL_KEYS = ["type", "version", "description", "stages", "reporting"]
    REQUIRED_STAGE_KEYS = ["description", "input_data", "reasoning_steps", "output_structure", "validation_rules"]
    REQUIRED_REPORTING_KEYS = ["template", "required_sections", "citation_requirements"]
    
    @staticmethod
    def validate_schema(schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a schema for completeness and structure.
        
        Args:
            schema: The schema dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check top-level keys
        for key in SchemaValidator.REQUIRED_TOP_LEVEL_KEYS:
            if key not in schema:
                errors.append(f"Missing required top-level key: {key}")
        
        # Validate stages
        if "stages" in schema:
            if not isinstance(schema["stages"], dict):
                errors.append("'stages' must be a dictionary")
            else:
                for stage_name, stage_data in schema["stages"].items():
                    stage_errors = SchemaValidator._validate_stage(stage_name, stage_data)
                    errors.extend(stage_errors)
        
        # Validate reporting section
        if "reporting" in schema:
            reporting_errors = SchemaValidator._validate_reporting(schema["reporting"])
            errors.extend(reporting_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_stage(stage_name: str, stage_data: Dict[str, Any]) -> List[str]:
        """
        Validate an individual stage structure.
        
        Args:
            stage_name: Name of the stage
            stage_data: Stage configuration dictionary
            
        Returns:
            List of error messages
        """
        errors = []
        
        for key in SchemaValidator.REQUIRED_STAGE_KEYS:
            if key not in stage_data:
                errors.append(f"Stage '{stage_name}': Missing required key '{key}'")
        
        # Validate input_data is a list
        if "input_data" in stage_data and not isinstance(stage_data["input_data"], list):
            errors.append(f"Stage '{stage_name}': 'input_data' must be a list")
        
        # Validate reasoning_steps is a list
        if "reasoning_steps" in stage_data and not isinstance(stage_data["reasoning_steps"], list):
            errors.append(f"Stage '{stage_name}': 'reasoning_steps' must be a list")
        
        # Validate output_structure is a dict
        if "output_structure" in stage_data and not isinstance(stage_data["output_structure"], dict):
            errors.append(f"Stage '{stage_name}': 'output_structure' must be a dictionary")
        
        # Validate validation_rules is a list
        if "validation_rules" in stage_data and not isinstance(stage_data["validation_rules"], list):
            errors.append(f"Stage '{stage_name}': 'validation_rules' must be a list")
        
        return errors
    
    @staticmethod
    def _validate_reporting(reporting_data: Dict[str, Any]) -> List[str]:
        """
        Validate reporting section structure.
        
        Args:
            reporting_data: Reporting configuration dictionary
            
        Returns:
            List of error messages
        """
        errors = []
        
        for key in SchemaValidator.REQUIRED_REPORTING_KEYS:
            if key not in reporting_data:
                errors.append(f"Reporting section: Missing required key '{key}'")
        
        # Validate required_sections is a list
        if "required_sections" in reporting_data and not isinstance(reporting_data["required_sections"], list):
            errors.append("Reporting section: 'required_sections' must be a list")
        
        return errors
    
    @staticmethod
    def get_schema_summary(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of a schema's structure.
        
        Args:
            schema: The schema to summarize
            
        Returns:
            Dictionary with schema summary information
        """
        summary = {
            "type": schema.get("type", "unknown"),
            "version": schema.get("version", "unknown"),
            "description": schema.get("description", ""),
            "stage_count": len(schema.get("stages", {})),
            "stages": list(schema.get("stages", {}).keys()),
            "required_sections": schema.get("reporting", {}).get("required_sections", [])
        }
        
        return summary
    
    @staticmethod
    def compare_schemas(schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two schemas and identify differences.
        
        Args:
            schema1: First schema
            schema2: Second schema
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "same_type": schema1.get("type") == schema2.get("type"),
            "stages_in_1_not_in_2": set(schema1.get("stages", {}).keys()) - set(schema2.get("stages", {}).keys()),
            "stages_in_2_not_in_1": set(schema2.get("stages", {}).keys()) - set(schema1.get("stages", {}).keys()),
            "common_stages": set(schema1.get("stages", {}).keys()) & set(schema2.get("stages", {}).keys())
        }
        
        return comparison


def validate_all_schemas():
    """
    Validate all schemas in the sgr_schemas module.
    
    Returns:
        Dictionary with validation results for each schema
    """
    from . import (
        RESOURCE_ASSESSMENT_SCHEMA,
        RISK_ANALYSIS_SCHEMA,
        ECONOMIC_VIABILITY_SCHEMA
    )
    
    schemas = {
        "resource_assessment": RESOURCE_ASSESSMENT_SCHEMA,
        "risk_analysis": RISK_ANALYSIS_SCHEMA,
        "economic_viability": ECONOMIC_VIABILITY_SCHEMA
    }
    
    results = {}
    
    for name, schema in schemas.items():
        is_valid, errors = SchemaValidator.validate_schema(schema)
        summary = SchemaValidator.get_schema_summary(schema)
        
        results[name] = {
            "valid": is_valid,
            "errors": errors,
            "summary": summary
        }
    
    return results


if __name__ == "__main__":
    """
    Run validation on all schemas when module is executed directly.
    """
    print("Validating SGR Schemas...")
    print("=" * 60)
    
    results = validate_all_schemas()
    
    for schema_name, result in results.items():
        print(f"\n{schema_name.upper()}")
        print("-" * 60)
        print(f"Valid: {result['valid']}")
        
        if result['errors']:
            print("\nErrors:")
            for error in result['errors']:
                print(f"  - {error}")
        else:
            print("[OK] No errors found")
        
        print(f"\nSummary:")
        print(f"  Type: {result['summary']['type']}")
        print(f"  Version: {result['summary']['version']}")
        print(f"  Stages: {result['summary']['stage_count']}")
        print(f"  Stage Names: {', '.join(result['summary']['stages'])}")
    
    print("\n" + "=" * 60)
    print("Validation complete!")

