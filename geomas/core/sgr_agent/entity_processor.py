"""
Entity Processor for geological analysis.

This module processes extracted entities and prepares them
for use in SGR-guided geological analysis.
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class EntityProcessor:
    """
    Processes and organizes extracted geological entities
    for structured analysis.
    """
    
    # Mapping of entity types to SGR schema stages
    ENTITY_SCHEMA_MAPPING = {
        "RESOURCE_POTENTIAL": ["resource_assessment", "economic_viability"],
        "ORE_COMPONENT": ["resource_assessment", "economic_viability"],
        "ORE_BODIES": ["resource_assessment"],
        "TECHNOLOGICAL": ["risk_analysis", "economic_viability"],
        "STRUCTURAL_TECTONIC": ["risk_analysis", "resource_assessment"],
        "GEO_CHEMICAL": ["resource_assessment"],
        "STUDY_INFO": ["resource_assessment", "risk_analysis", "economic_viability"],
        "INFO_SOURCES": ["resource_assessment", "risk_analysis", "economic_viability"],
        "GENERAL_INFO": ["resource_assessment"],
        "MINERALOGICAL": ["resource_assessment"],
        "ORE_FORMATION": ["resource_assessment"],
        "METALLOGENIC_CHAR": ["resource_assessment"],
        "FORMATION_CONDITIONS": ["risk_analysis"],
        "METASOMATIC": ["resource_assessment"],
        "STRATIGRAPHY": ["resource_assessment"],
        "GEODYNAMIC": ["risk_analysis"],
        "ORE_COMPOSITION": ["resource_assessment", "economic_viability"]
    }
    
    def __init__(self):
        """Initialize entity processor."""
        self.logger = logger
    
    def process_entities(
        self,
        entities: Dict[str, Any],
        schema_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process and organize entities for analysis.
        
        Args:
            entities: Raw entities from BERT NER
            schema_type: Target schema type for filtering
            
        Returns:
            Processed entities organized by relevance
        """
        if not entities or not entities.get("entities_by_type"):
            return {"processed_entities": {}, "entity_count": 0, "relevant_types": []}
        
        entities_by_type = entities["entities_by_type"]
        
        # Filter entities relevant to schema if specified
        if schema_type:
            relevant_entities = self._filter_by_schema(entities_by_type, schema_type)
        else:
            relevant_entities = entities_by_type
        
        # Deduplicate entities
        deduplicated = self._deduplicate_entities(relevant_entities)
        
        # Group by confidence
        confidence_groups = self._group_by_confidence(deduplicated)
        
        # Extract key information
        key_info = self._extract_key_information(deduplicated)
        
        return {
            "processed_entities": deduplicated,
            "confidence_groups": confidence_groups,
            "key_information": key_info,
            "entity_count": sum(len(v) for v in deduplicated.values()),
            "relevant_types": list(deduplicated.keys()),
            "schema_relevance": self._calculate_schema_relevance(deduplicated)
        }
    
    def _filter_by_schema(
        self,
        entities_by_type: Dict[str, List],
        schema_type: str
    ) -> Dict[str, List]:
        """Filter entities relevant to specific schema."""
        filtered = {}
        
        for entity_type, entity_list in entities_by_type.items():
            if entity_type in self.ENTITY_SCHEMA_MAPPING:
                relevant_schemas = self.ENTITY_SCHEMA_MAPPING[entity_type]
                if schema_type in relevant_schemas:
                    filtered[entity_type] = entity_list
        
        return filtered
    
    def _deduplicate_entities(
        self,
        entities_by_type: Dict[str, List]
    ) -> Dict[str, List]:
        """Remove duplicate entities (same text)."""
        deduplicated = {}
        
        for entity_type, entity_list in entities_by_type.items():
            seen_texts = set()
            unique_entities = []
            
            for entity in entity_list:
                text_lower = entity["text"].lower().strip()
                if text_lower not in seen_texts:
                    seen_texts.add(text_lower)
                    unique_entities.append(entity)
            
            deduplicated[entity_type] = unique_entities
        
        return deduplicated
    
    def _group_by_confidence(
        self,
        entities_by_type: Dict[str, List]
    ) -> Dict[str, Dict[str, List]]:
        """Group entities by confidence level."""
        confidence_groups = {
            "high": defaultdict(list),      # confidence >= 0.9
            "medium": defaultdict(list),    # 0.7 <= confidence < 0.9
            "low": defaultdict(list)        # confidence < 0.7
        }
        
        for entity_type, entity_list in entities_by_type.items():
            for entity in entity_list:
                confidence = entity.get("confidence", 0.0)
                
                if confidence >= 0.9:
                    confidence_groups["high"][entity_type].append(entity)
                elif confidence >= 0.7:
                    confidence_groups["medium"][entity_type].append(entity)
                else:
                    confidence_groups["low"][entity_type].append(entity)
        
        return {
            k: dict(v) for k, v in confidence_groups.items()
        }
    
    def _extract_key_information(
        self,
        entities_by_type: Dict[str, List]
    ) -> Dict[str, Any]:
        """Extract key information from entities."""
        key_info = {}
        
        # Extract resource information
        if "RESOURCE_POTENTIAL" in entities_by_type:
            key_info["resources"] = [
                e["text"] for e in entities_by_type["RESOURCE_POTENTIAL"]
            ]
        
        # Extract ore components
        if "ORE_COMPONENT" in entities_by_type:
            key_info["ore_components"] = [
                e["text"] for e in entities_by_type["ORE_COMPONENT"]
            ]
        
        # Extract location/general info
        if "GENERAL_INFO" in entities_by_type:
            key_info["general_info"] = [
                e["text"] for e in entities_by_type["GENERAL_INFO"]
            ][:5]  # Limit to top 5
        
        # Extract technological aspects
        if "TECHNOLOGICAL" in entities_by_type:
            key_info["technology"] = [
                e["text"] for e in entities_by_type["TECHNOLOGICAL"]
            ]
        
        return key_info
    
    def _calculate_schema_relevance(
        self,
        entities_by_type: Dict[str, List]
    ) -> Dict[str, float]:
        """Calculate relevance score for each schema type."""
        schema_scores = defaultdict(float)
        
        for entity_type, entity_list in entities_by_type.items():
            if entity_type in self.ENTITY_SCHEMA_MAPPING:
                relevant_schemas = self.ENTITY_SCHEMA_MAPPING[entity_type]
                score_per_entity = len(entity_list)
                
                for schema in relevant_schemas:
                    schema_scores[schema] += score_per_entity
        
        # Normalize scores
        if schema_scores:
            max_score = max(schema_scores.values())
            if max_score > 0:
                schema_scores = {
                    k: round(v / max_score, 2)
                    for k, v in schema_scores.items()
                }
        
        return dict(schema_scores)
    
    def create_entity_summary(
        self,
        processed_entities: Dict[str, Any]
    ) -> str:
        """
        Create human-readable summary of extracted entities.
        
        Args:
            processed_entities: Processed entities from process_entities()
            
        Returns:
            Formatted summary string
        """
        if not processed_entities or processed_entities["entity_count"] == 0:
            return "No entities extracted."
        
        summary_parts = [
            f"Extracted {processed_entities['entity_count']} entities across {len(processed_entities['relevant_types'])} types:"
        ]
        
        # List entity types with counts
        for entity_type in processed_entities["relevant_types"]:
            count = len(processed_entities["processed_entities"][entity_type])
            summary_parts.append(f"  - {entity_type}: {count}")
        
        # Add key information
        key_info = processed_entities.get("key_information", {})
        if key_info:
            summary_parts.append("\nKey Information:")
            
            if "ore_components" in key_info:
                summary_parts.append(f"  Ore Components: {', '.join(key_info['ore_components'][:5])}")
            
            if "resources" in key_info:
                summary_parts.append(f"  Resources: {', '.join(key_info['resources'][:3])}")
        
        # Add schema relevance
        schema_rel = processed_entities.get("schema_relevance", {})
        if schema_rel:
            summary_parts.append("\nRecommended Analysis Types:")
            sorted_schemas = sorted(schema_rel.items(), key=lambda x: x[1], reverse=True)
            for schema, score in sorted_schemas[:3]:
                summary_parts.append(f"  - {schema}: {score:.0%} relevance")
        
        return "\n".join(summary_parts)

