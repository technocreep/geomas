"""
Economic Viability Schema for SGR Deep Research.

This schema guides the structured assessment of economic potential
and financial viability of mineral deposits.
"""

ECONOMIC_VIABILITY_SCHEMA = {
    "type": "economic_viability_workflow",
    "version": "1.0",
    "description": "Comprehensive economic assessment of mineral deposit development potential",
    
    "stages": {
        "resource_valuation": {
            "description": "Calculate gross in-situ value of mineral resources",
            "input_data": [
                "resource_estimate",
                "commodity_prices",
                "metal_grades",
                "tonnage_by_category"
            ],
            "reasoning_steps": [
                "calculate_metal_content_by_category",
                "apply_current_commodity_prices",
                "calculate_gross_metal_value",
                "assess_value_sensitivity_to_price",
                "compare_to_industry_benchmarks"
            ],
            "output_structure": {
                "resource_value": {
                    "type": "object",
                    "properties": {
                        "gross_insitu_value": {
                            "total": {"type": "number", "unit": "USD"},
                            "per_tonne": {"type": "number", "unit": "USD/tonne"}
                        },
                        "value_by_category": {
                            "measured": {"type": "number", "unit": "USD"},
                            "indicated": {"type": "number", "unit": "USD"},
                            "inferred": {"type": "number", "unit": "USD"}
                        },
                        "value_by_commodity": {
                            "type": "object",
                            "additionalProperties": {"type": "number", "unit": "USD"}
                        },
                        "price_assumptions": {
                            "type": "object",
                            "additionalProperties": {"type": "number", "unit": "USD/unit"}
                        }
                    }
                }
            },
            "validation_rules": [
                "price_assumptions_documented",
                "metal_prices_within_market_range",
                "resource_categories_properly_valued",
                "by_product_credits_included"
            ]
        },
        
        "operating_cost_estimation": {
            "description": "Estimate life-of-mine operating costs and unit costs",
            "input_data": [
                "mining_method",
                "processing_flowsheet",
                "metallurgical_recovery",
                "production_rate",
                "labor_costs",
                "energy_costs",
                "consumables_costs"
            ],
            "reasoning_steps": [
                "estimate_mining_costs",
                "estimate_processing_costs",
                "estimate_general_and_administrative_costs",
                "calculate_unit_operating_costs",
                "benchmark_against_comparable_operations"
            ],
            "output_structure": {
                "operating_costs": {
                    "type": "object",
                    "properties": {
                        "mining_cost_per_tonne": {"type": "number", "unit": "USD/tonne"},
                        "processing_cost_per_tonne": {"type": "number", "unit": "USD/tonne"},
                        "g_and_a_cost_per_tonne": {"type": "number", "unit": "USD/tonne"},
                        "total_operating_cost": {
                            "per_tonne_ore": {"type": "number", "unit": "USD/tonne"},
                            "per_unit_product": {"type": "number", "unit": "USD/oz or USD/lb"}
                        },
                        "annual_operating_cost": {"type": "number", "unit": "USD/year"},
                        "cost_breakdown": {
                            "type": "object",
                            "properties": {
                                "mining": {"type": "number", "unit": "percent"},
                                "processing": {"type": "number", "unit": "percent"},
                                "g_and_a": {"type": "number", "unit": "percent"},
                                "other": {"type": "number", "unit": "percent"}
                            }
                        }
                    }
                }
            },
            "validation_rules": [
                "costs_benchmarked_against_peers",
                "unit_costs_within_industry_range",
                "cost_escalation_considered",
                "all_cost_categories_included"
            ]
        },
        
        "capital_cost_estimation": {
            "description": "Estimate initial and sustaining capital requirements",
            "input_data": [
                "mining_infrastructure",
                "processing_plant_design",
                "tailings_facility",
                "infrastructure_requirements",
                "equipment_list",
                "indirect_costs"
            ],
            "reasoning_steps": [
                "estimate_mine_development_capex",
                "estimate_processing_plant_capex",
                "estimate_infrastructure_capex",
                "calculate_indirect_costs_and_contingency",
                "estimate_sustaining_capital",
                "develop_capital_expenditure_schedule"
            ],
            "output_structure": {
                "capital_costs": {
                    "type": "object",
                    "properties": {
                        "initial_capital": {
                            "total": {"type": "number", "unit": "USD"},
                            "per_annual_tonne": {"type": "number", "unit": "USD/tpa"}
                        },
                        "capex_breakdown": {
                            "mine_development": {"type": "number", "unit": "USD"},
                            "processing_plant": {"type": "number", "unit": "USD"},
                            "infrastructure": {"type": "number", "unit": "USD"},
                            "indirect_costs": {"type": "number", "unit": "USD"},
                            "contingency": {"type": "number", "unit": "USD"}
                        },
                        "sustaining_capital": {
                            "annual_average": {"type": "number", "unit": "USD/year"},
                            "total_lom": {"type": "number", "unit": "USD"}
                        },
                        "capital_intensity": {"type": "number", "unit": "USD/annual_tonne"},
                        "estimate_accuracy": {
                            "type": "string",
                            "enum": ["scoping_±40%", "prefeasibility_±25%", "feasibility_±15%"]
                        }
                    }
                }
            },
            "validation_rules": [
                "all_major_capex_items_included",
                "contingency_appropriate_for_study_level",
                "sustaining_capital_estimated",
                "capex_benchmarked_against_peers"
            ]
        },
        
        "financial_modeling": {
            "description": "Develop comprehensive financial model and calculate key metrics",
            "input_data": [
                "resource_value",
                "operating_costs",
                "capital_costs",
                "production_schedule",
                "commodity_price_forecasts",
                "discount_rate",
                "taxation_regime"
            ],
            "reasoning_steps": [
                "build_production_schedule",
                "calculate_annual_revenues",
                "calculate_annual_costs",
                "apply_taxation_and_royalties",
                "calculate_free_cash_flows",
                "calculate_npv_and_irr",
                "calculate_payback_period"
            ],
            "output_structure": {
                "financial_metrics": {
                    "type": "object",
                    "properties": {
                        "npv": {
                            "at_5_percent": {"type": "number", "unit": "USD"},
                            "at_8_percent": {"type": "number", "unit": "USD"},
                            "at_10_percent": {"type": "number", "unit": "USD"}
                        },
                        "irr": {"type": "number", "unit": "percent"},
                        "payback_period": {"type": "number", "unit": "years"},
                        "mine_life": {"type": "number", "unit": "years"},
                        "peak_annual_production": {"type": "number", "unit": "tonnes/year"},
                        "average_annual_revenue": {"type": "number", "unit": "USD/year"},
                        "average_annual_ebitda": {"type": "number", "unit": "USD/year"},
                        "total_undiscounted_cash_flow": {"type": "number", "unit": "USD"}
                    }
                }
            },
            "validation_rules": [
                "financial_model_balanced",
                "all_cash_flows_accounted",
                "taxation_correctly_applied",
                "discount_rate_justified"
            ]
        },
        
        "sensitivity_analysis": {
            "description": "Analyze sensitivity to key variables and assess economic robustness",
            "input_data": [
                "financial_model",
                "commodity_price_range",
                "operating_cost_range",
                "capital_cost_range",
                "production_rate_scenarios",
                "grade_variability"
            ],
            "reasoning_steps": [
                "identify_key_value_drivers",
                "perform_one_way_sensitivity_analysis",
                "perform_scenario_analysis",
                "calculate_breakeven_values",
                "assess_downside_risk",
                "identify_upside_potential"
            ],
            "output_structure": {
                "sensitivity_results": {
                    "type": "object",
                    "properties": {
                        "price_sensitivity": {
                            "npv_change_per_percent": {"type": "number", "unit": "USD"},
                            "breakeven_price": {"type": "number", "unit": "USD/unit"}
                        },
                        "opex_sensitivity": {
                            "npv_change_per_percent": {"type": "number", "unit": "USD"}
                        },
                        "capex_sensitivity": {
                            "npv_change_per_percent": {"type": "number", "unit": "USD"}
                        },
                        "grade_sensitivity": {
                            "npv_change_per_percent": {"type": "number", "unit": "USD"}
                        },
                        "scenario_analysis": {
                            "base_case": {
                                "npv": {"type": "number", "unit": "USD"},
                                "irr": {"type": "number", "unit": "percent"}
                            },
                            "upside_case": {
                                "npv": {"type": "number", "unit": "USD"},
                                "irr": {"type": "number", "unit": "percent"},
                                "assumptions": {"type": "string"}
                            },
                            "downside_case": {
                                "npv": {"type": "number", "unit": "USD"},
                                "irr": {"type": "number", "unit": "percent"},
                                "assumptions": {"type": "string"}
                            }
                        },
                        "value_at_risk": {
                            "p10_npv": {"type": "number", "unit": "USD"},
                            "p50_npv": {"type": "number", "unit": "USD"},
                            "p90_npv": {"type": "number", "unit": "USD"}
                        }
                    }
                }
            },
            "validation_rules": [
                "key_sensitivities_identified",
                "breakeven_values_calculated",
                "scenario_assumptions_documented",
                "risk_distribution_quantified"
            ]
        },
        
        "investment_decision_framework": {
            "description": "Synthesize economic analysis into investment recommendation",
            "input_data": [
                "financial_metrics",
                "sensitivity_results",
                "risk_assessment",
                "market_outlook",
                "strategic_fit"
            ],
            "reasoning_steps": [
                "assess_economic_viability",
                "evaluate_risk_adjusted_returns",
                "compare_to_investment_hurdle_rates",
                "assess_competitive_position",
                "identify_value_creation_opportunities",
                "formulate_investment_recommendation"
            ],
            "output_structure": {
                "investment_assessment": {
                    "type": "object",
                    "properties": {
                        "economic_viability": {
                            "type": "string",
                            "enum": ["highly_attractive", "attractive", "marginal", "uneconomic"]
                        },
                        "investment_recommendation": {
                            "type": "string",
                            "enum": ["proceed_to_development", "advance_to_next_study", "continue_exploration", "divest", "monitor"]
                        },
                        "key_strengths": {"type": "array", "items": {"type": "string"}},
                        "key_weaknesses": {"type": "array", "items": {"type": "string"}},
                        "critical_success_factors": {"type": "array", "items": {"type": "string"}},
                        "next_steps": {
                            "type": "array",
                            "items": {
                                "action": {"type": "string"},
                                "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                                "timeline": {"type": "string"},
                                "estimated_cost": {"type": "number", "unit": "USD"}
                            }
                        }
                    }
                }
            },
            "validation_rules": [
                "recommendation_supported_by_analysis",
                "risks_and_opportunities_balanced",
                "next_steps_actionable",
                "investment_criteria_met"
            ]
        }
    },
    
    "reporting": {
        "template": "economic_assessment_report",
        "required_sections": [
            "executive_summary",
            "project_overview",
            "resource_valuation",
            "operating_cost_estimate",
            "capital_cost_estimate",
            "financial_model_and_metrics",
            "sensitivity_and_scenario_analysis",
            "economic_comparison_to_peers",
            "risks_and_opportunities",
            "conclusions_and_recommendations"
        ],
        "citation_requirements": {
            "sources_per_statement": "minimum_1",
            "preferred_sources": [
                "technical_studies",
                "market_reports",
                "peer_company_data",
                "economic_databases",
                "expert_estimates"
            ],
            "cost_estimate_traceability": "detailed_basis_of_estimate"
        },
        "visualization_requirements": [
            "npv_vs_discount_rate_chart",
            "tornado_diagram_sensitivity",
            "cashflow_waterfall",
            "peer_comparison_charts",
            "production_profile_chart"
        ]
    },
    
    "quality_checks": {
        "mandatory_validations": [
            "financial_model_mathematically_correct",
            "all_costs_and_revenues_included",
            "assumptions_documented_and_justified",
            "sensitivity_analysis_comprehensive",
            "peer_benchmarking_completed"
        ],
        "warning_triggers": [
            "npv_highly_sensitive_to_single_variable",
            "operating_costs_above_peer_median",
            "capital_intensity_very_high",
            "irr_below_hurdle_rate",
            "payback_period_exceeds_threshold"
        ]
    },
    
    "entity_mapping": {
        "description": "Mapping to geomas BERT NER entity types",
        "relevant_entities": [
            "RESOURCE_POTENTIAL",
            "ORE_COMPONENT",
            "TECHNOLOGICAL",
            "GENERAL_INFO",
            "INFO_SOURCES"
        ]
    }
}

