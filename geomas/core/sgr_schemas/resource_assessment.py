"""
Resource Assessment Schema for SGR Deep Research.

This schema guides the structured reasoning process for assessing
mineral resource potential of geological deposits.
"""

RESOURCE_ASSESSMENT_SCHEMA = {
    "type": "resource_assessment_workflow",
    "version": "1.0",
    "description": "Comprehensive mineral resource assessment following international standards (JORC, NI 43-101)",
    
    "stages": {
        "data_collection": {
            "description": "Gather and validate all available geological data for the deposit",
            "input_data": [
                "drill_hole_data",
                "assay_results",
                "geological_maps",
                "geophysical_surveys",
                "geochemical_samples",
                "historical_reports"
            ],
            "reasoning_steps": [
                "validate_data_completeness",
                "assess_data_quality_and_reliability",
                "identify_data_gaps",
                "cross_reference_multiple_sources"
            ],
            "output_structure": {
                "data_inventory": {
                    "type": "object",
                    "properties": {
                        "drill_holes_count": {"type": "integer"},
                        "assay_samples_count": {"type": "integer"},
                        "data_quality_rating": {"type": "string", "enum": ["high", "medium", "low"]},
                        "data_gaps": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "validation_rules": [
                "minimum_drill_hole_density",
                "assay_qaqc_standards_met",
                "spatial_coverage_adequacy"
            ]
        },
        
        "geological_interpretation": {
            "description": "Develop geological model and define mineralization domains",
            "input_data": [
                "drill_hole_logs",
                "structural_geology_data",
                "lithological_units",
                "alteration_zones",
                "mineralization_patterns"
            ],
            "reasoning_steps": [
                "identify_geological_domains",
                "define_mineralization_boundaries",
                "create_3d_geological_model",
                "validate_model_against_observations"
            ],
            "output_structure": {
                "geological_model": {
                    "type": "object",
                    "properties": {
                        "domains_count": {"type": "integer"},
                        "primary_ore_type": {"type": "string"},
                        "mineralization_style": {"type": "string"},
                        "structural_controls": {"type": "array", "items": {"type": "string"}},
                        "confidence_level": {"type": "string", "enum": ["high", "moderate", "low"]}
                    }
                }
            },
            "validation_rules": [
                "geological_continuity_validated",
                "domain_boundaries_justified",
                "model_honors_structural_controls"
            ]
        },
        
        "grade_estimation": {
            "description": "Estimate metal grades and tonnages for resource categories",
            "input_data": [
                "composite_samples",
                "geological_domains",
                "grade_distribution_analysis",
                "spatial_statistics"
            ],
            "reasoning_steps": [
                "perform_exploratory_data_analysis",
                "analyze_grade_distribution",
                "select_estimation_methodology",
                "calculate_block_grades",
                "classify_resources_by_confidence"
            ],
            "output_structure": {
                "resource_estimate": {
                    "type": "object",
                    "properties": {
                        "measured_resources": {
                            "tonnage": {"type": "number", "unit": "tonnes"},
                            "grade": {"type": "number", "unit": "g/t or %"},
                            "metal_content": {"type": "number", "unit": "kg or oz"}
                        },
                        "indicated_resources": {
                            "tonnage": {"type": "number", "unit": "tonnes"},
                            "grade": {"type": "number", "unit": "g/t or %"},
                            "metal_content": {"type": "number", "unit": "kg or oz"}
                        },
                        "inferred_resources": {
                            "tonnage": {"type": "number", "unit": "tonnes"},
                            "grade": {"type": "number", "unit": "g/t or %"},
                            "metal_content": {"type": "number", "unit": "kg or oz"}
                        },
                        "cutoff_grade": {"type": "number", "unit": "g/t or %"},
                        "estimation_method": {"type": "string"}
                    }
                }
            },
            "validation_rules": [
                "grade_tonnage_relationship_validated",
                "estimation_parameters_justified",
                "classification_criteria_documented",
                "qaqc_checks_passed"
            ]
        },
        
        "uncertainty_analysis": {
            "description": "Quantify and document estimation uncertainties",
            "input_data": [
                "resource_estimate",
                "drill_hole_spacing",
                "grade_variability",
                "geological_complexity"
            ],
            "reasoning_steps": [
                "assess_data_density_impact",
                "evaluate_grade_continuity",
                "analyze_estimation_variance",
                "quantify_classification_confidence"
            ],
            "output_structure": {
                "uncertainty_metrics": {
                    "type": "object",
                    "properties": {
                        "estimation_precision": {"type": "number", "unit": "percent"},
                        "confidence_intervals": {
                            "type": "object",
                            "properties": {
                                "p10": {"type": "number"},
                                "p50": {"type": "number"},
                                "p90": {"type": "number"}
                            }
                        },
                        "key_risks": {"type": "array", "items": {"type": "string"}},
                        "sensitivity_factors": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "validation_rules": [
                "uncertainty_quantified_for_all_categories",
                "sensitivity_analysis_completed",
                "risk_factors_documented"
            ]
        }
    },
    
    "reporting": {
        "template": "resource_estimate_technical_report",
        "required_sections": [
            "executive_summary",
            "property_description_and_location",
            "geological_setting",
            "deposit_type_and_mineralization",
            "exploration_and_drilling",
            "sample_preparation_and_analysis",
            "data_verification",
            "geological_interpretation",
            "resource_estimation_methodology",
            "resource_estimate_statement",
            "uncertainty_and_classification",
            "recommendations"
        ],
        "citation_requirements": {
            "sources_per_statement": "minimum_1",
            "preferred_sources": [
                "drill_hole_database",
                "assay_certificates",
                "geological_maps",
                "technical_reports",
                "published_literature"
            ],
            "data_traceability": "full_chain_of_custody"
        },
        "compliance_standards": [
            "JORC_Code_2012",
            "NI_43-101",
            "SAMREC_Code"
        ]
    },
    
    "quality_checks": {
        "mandatory_validations": [
            "resource_categories_properly_classified",
            "estimation_methodology_appropriate",
            "cutoff_grade_justified",
            "mining_modifying_factors_considered",
            "competent_person_review_completed"
        ],
        "warning_triggers": [
            "high_coefficient_of_variation",
            "insufficient_drill_spacing",
            "incomplete_qaqc_data",
            "geological_interpretation_uncertainty"
        ]
    },
    
    "entity_mapping": {
        "description": "Mapping to geomas BERT NER entity types",
        "relevant_entities": [
            "RESOURCE_POTENTIAL",
            "ORE_COMPONENT",
            "ORE_BODIES",
            "GEOLOGICAL_SETTING",
            "STUDY_INFO",
            "TECHNOLOGICAL"
        ]
    }
}

