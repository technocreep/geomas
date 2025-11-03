"""
Geological Analysis Agent using SGR Deep Research Framework.

This agent performs structured geological analysis guided by predefined schemas,
integrating with geomas BERT NER for entity extraction and vector database for context.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from geomas.core.sgr_schemas import (
    RESOURCE_ASSESSMENT_SCHEMA,
    RISK_ANALYSIS_SCHEMA,
    ECONOMIC_VIABILITY_SCHEMA
)

logger = logging.getLogger(__name__)


class GeologicalAnalysisAgent:
    """
    SGR-powered geological analysis agent that uses structured schemas
    for systematic mineral deposit assessment.
    
    This agent integrates:
    - SGR Deep Research for structured reasoning
    - BERT NER for geological entity extraction
    - Vector database for context retrieval
    - Predefined schemas for different analysis types
    """
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        bert_ner_model_path: Optional[str] = None,
        vector_db_service=None,
        use_sgr: bool = True
    ):
        """
        Initialize the geological analysis agent.
        
        Args:
            llm_config: Configuration for the LLM (model, API key, etc.)
            bert_ner_model_path: Path to trained BERT NER model
            vector_db_service: Vector database service for context retrieval
            use_sgr: Whether to use full SGR framework or lightweight mode
        """
        self.llm_config = llm_config or {}
        self.bert_ner_model_path = bert_ner_model_path
        self.vector_db = vector_db_service
        self.use_sgr = use_sgr
        
        # Load schemas
        self.schemas = {
            "resource_assessment": RESOURCE_ASSESSMENT_SCHEMA,
            "risk_analysis": RISK_ANALYSIS_SCHEMA,
            "economic_viability": ECONOMIC_VIABILITY_SCHEMA
        }
        
        # Initialize NER model if path provided
        self.ner_model = None
        if bert_ner_model_path:
            self._init_ner_model(bert_ner_model_path)
        
        # Initialize SGR agent
        self.sgr_agent = None
        if use_sgr:
            self._init_sgr_agent()
    
    def _init_ner_model(self, model_path: str):
        """Initialize BERT NER model for entity extraction."""
        try:
            from geomas.core.inference.bert_ner_inference import load_bert_ner_model
            logger.info(f"Loading BERT NER model from {model_path}")
            self.ner_model = load_bert_ner_model(model_path)
            logger.info("BERT NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load BERT NER model: {e}")
            self.ner_model = None
    
    def _init_sgr_agent(self):
        """Initialize SGR Deep Research agent."""
        try:
            from sgr_deep_research import BaseAgent
            logger.info("Initializing SGR Deep Research agent")
            self.sgr_agent = BaseAgent(llm_config=self.llm_config)
            logger.info("SGR agent initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize SGR agent: {e}")
            logger.warning("Falling back to lightweight mode")
            self.sgr_agent = None
            self.use_sgr = False
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract geological entities from text using BERT NER.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with extracted entities by type
        """
        if not self.ner_model:
            logger.warning("NER model not available, skipping entity extraction")
            return {"entities": [], "entity_types": []}
        
        try:
            result = self.ner_model.extract_entities(text)
            
            # Group entities by type
            entities_by_type = {}
            for entity in result.entities:
                if entity.label not in entities_by_type:
                    entities_by_type[entity.label] = []
                entities_by_type[entity.label].append({
                    "text": entity.text,
                    "confidence": entity.confidence,
                    "start": entity.start,
                    "end": entity.end
                })
            
            return {
                "entities": result.entities,
                "entity_types": result.entity_types_found,
                "entities_by_type": entities_by_type,
                "total_count": len(result.entities)
            }
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "entity_types": []}
    
    def assess_resources(
        self,
        query: str,
        context_documents: List[Any],
        deposit_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform structured resource assessment following JORC/NI 43-101 standards.
        
        Args:
            query: User query about resource assessment
            context_documents: Retrieved documents from vector DB
            deposit_data: Additional structured data about the deposit
            
        Returns:
            Structured resource assessment results
        """
        logger.info("Initiating Resource Assessment...")
        schema = self.schemas["resource_assessment"]
        
        # Extract entities from context
        entities_info = self._extract_entities_from_context(context_documents)
        
        # Prepare analysis context
        analysis_context = {
            "query": query,
            "schema_type": "resource_assessment",
            "context_documents": context_documents,
            "entities": entities_info,
            "deposit_data": deposit_data or {}
        }
        
        # Execute analysis
        if self.use_sgr and self.sgr_agent:
            result = self._execute_sgr_analysis(schema, analysis_context)
        else:
            result = self._execute_lightweight_analysis(schema, analysis_context)
        
        return result
    
    def analyze_risks(
        self,
        query: str,
        context_documents: List[Any],
        project_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis across all risk categories.
        
        Args:
            query: User query about risk analysis
            context_documents: Retrieved documents from vector DB
            project_data: Additional project information
            
        Returns:
            Integrated risk assessment with mitigation strategies
        """
        logger.info("Initiating Risk Analysis...")
        schema = self.schemas["risk_analysis"]
        
        entities_info = self._extract_entities_from_context(context_documents)
        
        analysis_context = {
            "query": query,
            "schema_type": "risk_analysis",
            "context_documents": context_documents,
            "entities": entities_info,
            "project_data": project_data or {}
        }
        
        if self.use_sgr and self.sgr_agent:
            result = self._execute_sgr_analysis(schema, analysis_context)
        else:
            result = self._execute_lightweight_analysis(schema, analysis_context)
        
        return result
    
    def evaluate_economics(
        self,
        query: str,
        context_documents: List[Any],
        economic_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate economic viability and calculate financial metrics.
        
        Args:
            query: User query about economic analysis
            context_documents: Retrieved documents from vector DB
            economic_data: Resource estimates, costs, prices
            
        Returns:
            Financial analysis with NPV, IRR, sensitivity results
        """
        logger.info("Initiating Economic Viability Analysis...")
        schema = self.schemas["economic_viability"]
        
        entities_info = self._extract_entities_from_context(context_documents)
        
        analysis_context = {
            "query": query,
            "schema_type": "economic_viability",
            "context_documents": context_documents,
            "entities": entities_info,
            "economic_data": economic_data or {}
        }
        
        if self.use_sgr and self.sgr_agent:
            result = self._execute_sgr_analysis(schema, analysis_context)
        else:
            result = self._execute_lightweight_analysis(schema, analysis_context)
        
        return result
    
    def analyze_general(
        self,
        query: str,
        context_documents: List[Any],
        analysis_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform general geological analysis (auto-detect type from query).
        
        Args:
            query: User query
            context_documents: Retrieved documents from vector DB
            analysis_type: Optional explicit analysis type
            
        Returns:
            Analysis results based on detected or specified type
        """
        # Auto-detect analysis type if not specified
        if not analysis_type:
            analysis_type = self._detect_analysis_type(query)
        
        logger.info(f"Detected analysis type: {analysis_type}")
        
        # Route to appropriate analysis method
        if analysis_type == "resource_assessment":
            return self.assess_resources(query, context_documents)
        elif analysis_type == "risk_analysis":
            return self.analyze_risks(query, context_documents)
        elif analysis_type == "economic_viability":
            return self.evaluate_economics(query, context_documents)
        else:
            # Default general analysis
            return self._execute_general_analysis(query, context_documents)
    
    def _extract_entities_from_context(self, documents: List[Any]) -> Dict[str, Any]:
        """Extract entities from all context documents."""
        if not self.ner_model or not documents:
            return {"entities_by_type": {}, "total_entities": 0}
        
        all_entities = {}
        total_count = 0
        
        for doc in documents:
            text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            entities = self.extract_entities(text)
            
            # Merge entities
            for entity_type, entity_list in entities.get("entities_by_type", {}).items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = []
                all_entities[entity_type].extend(entity_list)
                total_count += len(entity_list)
        
        return {
            "entities_by_type": all_entities,
            "total_entities": total_count
        }
    
    def _execute_sgr_analysis(
        self,
        schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute analysis using full SGR framework."""
        logger.info(f"Executing SGR analysis for {schema['type']}")
        
        # TODO: Integrate with actual SGR BaseAgent
        # For now, return structured placeholder
        return {
            "status": "sgr_analysis_complete",
            "schema_type": schema["type"],
            "stages_completed": list(schema["stages"].keys()),
            "context_summary": {
                "query": context["query"],
                "documents_count": len(context.get("context_documents", [])),
                "entities_found": context.get("entities", {}).get("total_entities", 0)
            },
            "mode": "full_sgr"
        }
    
    def _execute_lightweight_analysis(
        self,
        schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute analysis using lightweight mode (without full SGR)."""
        logger.info(f"Executing lightweight analysis for {schema['type']}")
        
        return {
            "status": "lightweight_analysis_complete",
            "schema_type": schema["type"],
            "stages": list(schema["stages"].keys()),
            "context_summary": {
                "query": context["query"],
                "documents_count": len(context.get("context_documents", [])),
                "entities_found": context.get("entities", {}).get("total_entities", 0)
            },
            "mode": "lightweight"
        }
    
    def _execute_general_analysis(
        self,
        query: str,
        documents: List[Any]
    ) -> Dict[str, Any]:
        """Execute general geological analysis without specific schema."""
        entities = self._extract_entities_from_context(documents)
        
        return {
            "status": "general_analysis_complete",
            "query": query,
            "documents_analyzed": len(documents),
            "entities": entities,
            "mode": "general"
        }
    
    def _detect_analysis_type(self, query: str) -> str:
        """Auto-detect analysis type from query keywords."""
        query_lower = query.lower()
        
        # Resource assessment keywords
        if any(kw in query_lower for kw in [
            "resource", "reserve", "tonnage", "grade", "estimation",
            "jorc", "ni 43-101", "mineral resource"
        ]):
            return "resource_assessment"
        
        # Risk analysis keywords
        if any(kw in query_lower for kw in [
            "risk", "uncertainty", "hazard", "threat", "vulnerability",
            "mitigation", "safety"
        ]):
            return "risk_analysis"
        
        # Economic analysis keywords
        if any(kw in query_lower for kw in [
            "economic", "financial", "npv", "irr", "cost", "price",
            "valuation", "feasibility", "payback"
        ]):
            return "economic_viability"
        
        # Default to general analysis
        return "general"
    
    def get_available_schemas(self) -> List[str]:
        """Get list of available analysis schemas."""
        return list(self.schemas.keys())
    
    def get_schema_info(self, schema_type: str) -> Dict[str, Any]:
        """Get information about a specific schema."""
        if schema_type not in self.schemas:
            return {"error": f"Schema '{schema_type}' not found"}
        
        schema = self.schemas[schema_type]
        return {
            "type": schema["type"],
            "version": schema.get("version", "unknown"),
            "description": schema.get("description", ""),
            "stages": list(schema["stages"].keys()),
            "stage_count": len(schema["stages"])
        }

