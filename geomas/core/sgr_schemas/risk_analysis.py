"""
Risk Analysis Schema for SGR Deep Research.

This schema guides the structured assessment of geological,
technical, and operational risks associated with mineral deposits.
"""

RISK_ANALYSIS_SCHEMA = {
    "type": "risk_analysis_workflow",
    "version": "1.0",
    "description": "Comprehensive risk assessment for mineral exploration and development projects",
    
    "stages": {
        "geological_risk_assessment": {
            "description": "Evaluate risks related to geological uncertainty and resource variability",
            "input_data": [
                "resource_estimate_confidence",
                "drill_hole_spacing",
                "geological_complexity",
                "grade_continuity_analysis",
                "structural_controls"
            ],
            "reasoning_steps": [
                "assess_data_coverage_adequacy",
                "evaluate_geological_continuity",
                "analyze_grade_variability",
                "identify_structural_complexities",
                "quantify_resource_confidence_levels"
            ],
            "output_structure": {
                "geological_risks": {
                    "type": "object",
                    "properties": {
                        "resource_confidence": {
                            "type": "string",
                            "enum": ["high", "moderate", "low"]
                        },
                        "grade_variability_risk": {
                            "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                            "likelihood": {"type": "string", "enum": ["very_likely", "likely", "possible", "unlikely"]},
                            "impact": {"type": "string"}
                        },
                        "geological_continuity_risk": {
                            "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                            "description": {"type": "string"}
                        },
                        "structural_complexity_risk": {
                            "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                            "factors": {"type": "array", "items": {"type": "string"}}
                        },
                        "key_uncertainties": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            },
            "validation_rules": [
                "all_risk_categories_evaluated",
                "severity_and_likelihood_justified",
                "mitigation_strategies_identified"
            ]
        },
        
        "technical_risk_assessment": {
            "description": "Evaluate technical and operational risks in mining and processing",
            "input_data": [
                "ore_characteristics",
                "metallurgical_test_results",
                "mining_method_considerations",
                "processing_complexity",
                "infrastructure_requirements"
            ],
            "reasoning_steps": [
                "assess_ore_processing_challenges",
                "evaluate_mining_method_suitability",
                "identify_metallurgical_risks",
                "assess_infrastructure_gaps",
                "evaluate_technology_maturity"
            ],
            "output_structure": {
                "technical_risks": {
                    "type": "object",
                    "properties": {
                        "mining_risks": {
                            "type": "array",
                            "items": {
                                "risk_name": {"type": "string"},
                                "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                                "mitigation": {"type": "string"}
                            }
                        },
                        "processing_risks": {
                            "type": "array",
                            "items": {
                                "risk_name": {"type": "string"},
                                "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                                "impact_description": {"type": "string"},
                                "mitigation": {"type": "string"}
                            }
                        },
                        "infrastructure_risks": {
                            "type": "array",
                            "items": {
                                "category": {"type": "string"},
                                "description": {"type": "string"},
                                "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]}
                            }
                        },
                        "overall_technical_risk_rating": {
                            "type": "string",
                            "enum": ["critical", "high", "moderate", "low"]
                        }
                    }
                }
            },
            "validation_rules": [
                "mining_method_feasibility_confirmed",
                "metallurgical_recovery_ranges_established",
                "infrastructure_requirements_documented",
                "technical_risks_prioritized"
            ]
        },
        
        "environmental_and_social_risk": {
            "description": "Assess environmental, social, and regulatory risks",
            "input_data": [
                "environmental_baseline_data",
                "regulatory_framework",
                "community_relations",
                "protected_areas",
                "water_resources"
            ],
            "reasoning_steps": [
                "identify_environmental_sensitivities",
                "assess_regulatory_compliance_requirements",
                "evaluate_social_license_factors",
                "analyze_permitting_risks",
                "assess_closure_and_rehabilitation_obligations"
            ],
            "output_structure": {
                "esg_risks": {
                    "type": "object",
                    "properties": {
                        "environmental_risks": {
                            "type": "array",
                            "items": {
                                "category": {"type": "string"},
                                "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                                "regulatory_requirements": {"type": "string"},
                                "mitigation_measures": {"type": "string"}
                            }
                        },
                        "social_risks": {
                            "type": "array",
                            "items": {
                                "stakeholder_group": {"type": "string"},
                                "concern": {"type": "string"},
                                "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                                "engagement_strategy": {"type": "string"}
                            }
                        },
                        "regulatory_risks": {
                            "permitting_complexity": {"type": "string", "enum": ["high", "moderate", "low"]},
                            "timeline_uncertainty": {"type": "string"},
                            "key_approvals_required": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            "validation_rules": [
                "environmental_baseline_adequate",
                "regulatory_framework_understood",
                "stakeholder_engagement_planned",
                "closure_costs_estimated"
            ]
        },
        
        "market_and_economic_risk": {
            "description": "Evaluate commodity price, market, and economic risks",
            "input_data": [
                "commodity_price_forecasts",
                "market_demand_analysis",
                "operating_cost_estimates",
                "capital_cost_estimates",
                "economic_sensitivity_analysis"
            ],
            "reasoning_steps": [
                "analyze_commodity_price_volatility",
                "assess_market_demand_trends",
                "evaluate_cost_estimate_reliability",
                "perform_economic_sensitivity_analysis",
                "identify_value_drivers_and_risks"
            ],
            "output_structure": {
                "economic_risks": {
                    "type": "object",
                    "properties": {
                        "commodity_price_risk": {
                            "volatility": {"type": "string", "enum": ["high", "moderate", "low"]},
                            "price_sensitivity": {"type": "number", "unit": "percent_change_in_npv"},
                            "downside_scenarios": {"type": "array", "items": {"type": "object"}}
                        },
                        "operating_cost_risk": {
                            "cost_uncertainty": {"type": "string", "enum": ["high", "moderate", "low"]},
                            "key_cost_drivers": {"type": "array", "items": {"type": "string"}},
                            "inflation_sensitivity": {"type": "string"}
                        },
                        "capital_cost_risk": {
                            "estimate_accuracy": {"type": "string", "enum": ["scoping", "prefeasibility", "feasibility"]},
                            "contingency_percentage": {"type": "number"},
                            "major_cost_components": {"type": "array", "items": {"type": "object"}}
                        },
                        "overall_economic_viability": {
                            "base_case_npv": {"type": "number", "unit": "USD"},
                            "breakeven_price": {"type": "number", "unit": "USD/unit"},
                            "risk_adjusted_return": {"type": "number", "unit": "percent"}
                        }
                    }
                }
            },
            "validation_rules": [
                "price_assumptions_benchmarked",
                "cost_estimates_peer_reviewed",
                "sensitivity_analysis_comprehensive",
                "economic_model_validated"
            ]
        },
        
        "integrated_risk_matrix": {
            "description": "Synthesize all risk categories into integrated risk assessment",
            "input_data": [
                "geological_risks",
                "technical_risks",
                "esg_risks",
                "economic_risks"
            ],
            "reasoning_steps": [
                "aggregate_risk_scores",
                "identify_critical_risk_interactions",
                "prioritize_risks_by_impact",
                "develop_integrated_mitigation_strategy",
                "establish_risk_monitoring_framework"
            ],
            "output_structure": {
                "integrated_risk_profile": {
                    "type": "object",
                    "properties": {
                        "overall_project_risk_rating": {
                            "type": "string",
                            "enum": ["critical", "high", "moderate", "low"]
                        },
                        "top_risks": {
                            "type": "array",
                            "maxItems": 10,
                            "items": {
                                "rank": {"type": "integer"},
                                "risk_name": {"type": "string"},
                                "category": {"type": "string"},
                                "severity": {"type": "string"},
                                "likelihood": {"type": "string"},
                                "risk_score": {"type": "number"},
                                "mitigation_priority": {"type": "string", "enum": ["immediate", "short_term", "long_term"]}
                            }
                        },
                        "risk_mitigation_plan": {
                            "type": "array",
                            "items": {
                                "risk_id": {"type": "string"},
                                "mitigation_actions": {"type": "array", "items": {"type": "string"}},
                                "responsible_party": {"type": "string"},
                                "timeline": {"type": "string"},
                                "cost_estimate": {"type": "number", "unit": "USD"}
                            }
                        },
                        "residual_risk_level": {
                            "type": "string",
                            "enum": ["acceptable", "manageable", "high", "unacceptable"]
                        }
                    }
                }
            },
            "validation_rules": [
                "all_risk_categories_integrated",
                "risk_interactions_identified",
                "mitigation_strategies_defined",
                "monitoring_framework_established",
                "residual_risks_acceptable"
            ]
        }
    },
    
    "reporting": {
        "template": "comprehensive_risk_assessment_report",
        "required_sections": [
            "executive_summary",
            "risk_assessment_methodology",
            "geological_risks",
            "technical_risks",
            "environmental_and_social_risks",
            "market_and_economic_risks",
            "integrated_risk_matrix",
            "risk_mitigation_strategies",
            "monitoring_and_review_plan",
            "conclusions_and_recommendations"
        ],
        "citation_requirements": {
            "sources_per_statement": "minimum_1",
            "preferred_sources": [
                "technical_studies",
                "geological_reports",
                "market_analysis",
                "regulatory_documents",
                "expert_opinions"
            ],
            "risk_documentation": "comprehensive_with_evidence"
        },
        "visualization_requirements": [
            "risk_matrix_heatmap",
            "tornado_diagram_sensitivity",
            "risk_score_distribution",
            "mitigation_timeline_gantt"
        ]
    },
    
    "quality_checks": {
        "mandatory_validations": [
            "all_risk_categories_assessed",
            "severity_and_likelihood_quantified",
            "mitigation_strategies_defined",
            "residual_risks_identified",
            "risk_owner_assigned"
        ],
        "warning_triggers": [
            "critical_risks_without_mitigation",
            "incomplete_risk_assessment",
            "insufficient_data_for_evaluation",
            "high_uncertainty_in_key_assumptions"
        ]
    },
    
    "entity_mapping": {
        "description": "Mapping to geomas BERT NER entity types",
        "relevant_entities": [
            "GENERAL_INFO",
            "STUDY_INFO",
            "TECHNOLOGICAL",
            "FORMATION_CONDITIONS",
            "STRUCTURAL_TECTONIC",
            "INFO_SOURCES"
        ]
    }
}

