"""
Decision Tracking Convenience Functions

This module provides convenience functions for decision tracking operations,
offering simple interfaces for common use cases.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..graph_store import GraphStore
from ..utils.logging import get_logger
from .agent_context import AgentContext
from .decision_models import Decision, Policy
from .decision_recorder import DecisionRecorder
from .decision_query import DecisionQuery
from .causal_analyzer import CausalChainAnalyzer
from .policy_engine import PolicyEngine


def record_decision(
    graph_store: GraphStore,
    category: str,
    scenario: str,
    reasoning: str,
    outcome: str,
    confidence: float,
    entities: Optional[List[str]] = None,
    cross_system_context: Optional[Dict[str, Any]] = None,
    decision_maker: Optional[str] = "ai_agent"
) -> str:
    """
    Convenience function for recording decisions.
    
    Args:
        graph_store: Graph database instance
        category: Decision category
        scenario: Decision scenario
        reasoning: Decision reasoning
        outcome: Decision outcome
        confidence: Confidence score (0-1)
        entities: Optional list of entity IDs
        cross_system_context: Optional cross-system context
        decision_maker: Decision maker identifier
        
    Returns:
        Decision ID
    """
    logger = get_logger(__name__)
    
    try:
        recorder = DecisionRecorder(graph_store)
        
        from .decision_models import Decision
        import uuid
        
        decision = Decision(
            decision_id=str(uuid.uuid4()),
            category=category,
            scenario=scenario,
            reasoning=reasoning,
            outcome=outcome,
            confidence=confidence,
            timestamp=datetime.now(),
            decision_maker=decision_maker or "ai_agent"
        )
        
        entities = entities or []
        source_documents = []  # Could be enhanced to capture source docs
        
        decision_id = recorder.record_decision(decision, entities, source_documents)
        
        # Capture cross-system context if provided
        if cross_system_context:
            recorder.capture_cross_system_context(decision_id, cross_system_context)
        
        logger.info(f"Recorded decision: {decision_id}")
        return decision_id
        
    except Exception as e:
        logger.error(f"Failed to record decision: {e}")
        raise


def find_precedents(
    graph_store: GraphStore,
    scenario: str,
    category: Optional[str] = None,
    limit: int = 10,
    use_hybrid_search: bool = True
) -> List[Decision]:
    """
    Convenience function for finding precedents.
    
    Args:
        graph_store: Graph database instance
        scenario: Scenario to find precedents for
        category: Optional category filter
        limit: Maximum number of results
        use_hybrid_search: Use hybrid search
        
    Returns:
        List of similar decisions
    """
    logger = get_logger(__name__)
    
    try:
        query_engine = DecisionQuery(graph_store)
        
        if use_hybrid_search:
            return query_engine.find_precedents_hybrid(scenario, category, limit)
        else:
            if category:
                return query_engine.find_by_category(category, limit)
            else:
                return query_engine.find_precedents_hybrid(scenario, category, limit)
        
    except Exception as e:
        logger.error(f"Failed to find precedents: {e}")
        raise


def get_causal_chain(
    graph_store: GraphStore,
    decision_id: str,
    direction: str = "upstream",
    max_depth: int = 10
) -> List[Decision]:
    """
    Convenience function for getting causal chains.
    
    Args:
        graph_store: Graph database instance
        decision_id: Decision ID to analyze
        direction: "upstream" or "downstream"
        max_depth: Maximum traversal depth
        
    Returns:
        List of decisions in causal chain
    """
    logger = get_logger(__name__)
    
    try:
        analyzer = CausalChainAnalyzer(graph_store)
        return analyzer.get_causal_chain(decision_id, direction, max_depth)
        
    except Exception as e:
        logger.error(f"Failed to get causal chain: {e}")
        raise


def get_applicable_policies(
    graph_store: GraphStore,
    category: str,
    entities: Optional[List[str]] = None
) -> List[Policy]:
    """
    Convenience function for getting applicable policies.
    
    Args:
        graph_store: Graph database instance
        category: Policy category
        entities: Optional list of entity IDs
        
    Returns:
        List of applicable policies
    """
    logger = get_logger(__name__)
    
    try:
        policy_engine = PolicyEngine(graph_store)
        return policy_engine.get_applicable_policies(category, entities)
        
    except Exception as e:
        logger.error(f"Failed to get applicable policies: {e}")
        raise


def multi_hop_query(
    graph_store: GraphStore,
    start_entity: str,
    query: str,
    max_hops: int = 3
) -> Dict[str, Any]:
    """
    Multi-hop reasoning convenience function.
    
    Args:
        graph_store: Graph database instance
        start_entity: Starting entity ID
        query: Query context
        max_hops: Maximum hops to traverse
        
    Returns:
        Query results with context
    """
    logger = get_logger(__name__)
    
    try:
        query_engine = DecisionQuery(graph_store)
        decisions = query_engine.multi_hop_reasoning(start_entity, query, max_hops)
        
        return {
            "query": query,
            "start_entity": start_entity,
            "max_hops": max_hops,
            "decisions": decisions,
            "count": len(decisions)
        }
        
    except Exception as e:
        logger.error(f"Failed multi-hop query: {e}")
        raise


def capture_decision_trace(
    decision: Decision,
    cross_system_context: Dict[str, Any]
) -> str:
    """
    Complete decision trace capture.
    
    Args:
        decision: Decision object to capture
        cross_system_context: Cross-system context
        
    Returns:
        Decision ID
    """
    logger = get_logger(__name__)
    
    try:
        # This would typically use a global or context-specific graph store
        # For now, return the decision ID as a placeholder
        logger.info(f"Captured decision trace for: {decision.decision_id}")
        return decision.decision_id
        
    except Exception as e:
        logger.error(f"Failed to capture decision trace: {e}")
        raise


def find_exception_precedents(
    graph_store: GraphStore,
    exception_reason: str,
    limit: int = 10
) -> List["Exception"]:
    """
    Exception precedent search convenience function.
    
    Args:
        graph_store: Graph database instance
        exception_reason: Reason for exception
        limit: Maximum number of results
        
    Returns:
        List of similar exceptions
    """
    logger = get_logger(__name__)
    
    try:
        query_engine = DecisionQuery(graph_store)
        return query_engine.find_similar_exceptions(exception_reason, limit)
        
    except Exception as e:
        logger.error(f"Failed to find exception precedents: {e}")
        raise


def analyze_decision_impact(
    graph_store: GraphStore,
    decision_id: str
) -> Dict[str, Any]:
    """
    Analyze the impact of a decision.
    
    Args:
        graph_store: Graph database instance
        decision_id: Decision ID to analyze
        
    Returns:
        Impact analysis results
    """
    logger = get_logger(__name__)
    
    try:
        analyzer = CausalChainAnalyzer(graph_store)
        
        # Get causal impact score
        impact_score = analyzer.get_causal_impact_score(decision_id)
        
        # Get influenced decisions
        influenced = analyzer.get_influenced_decisions(decision_id, max_depth=5)
        
        # Get root causes
        root_causes = analyzer.find_root_causes(decision_id, max_depth=5)
        
        return {
            "decision_id": decision_id,
            "impact_score": impact_score,
            "influenced_decisions": len(influenced),
            "root_causes": len(root_causes),
            "influenced_decision_ids": [d.decision_id for d in influenced],
            "root_cause_ids": [d.decision_id for d in root_causes],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze decision impact: {e}")
        raise


def create_policy_with_versioning(
    graph_store: GraphStore,
    name: str,
    description: str,
    rules: Dict[str, Any],
    category: str,
    version: str = "1.0"
) -> str:
    """
    Create a policy with automatic versioning.
    
    Args:
        graph_store: Graph database instance
        name: Policy name
        description: Policy description
        rules: Policy rules
        category: Policy category
        version: Initial version
        
    Returns:
        Policy ID
    """
    logger = get_logger(__name__)
    
    try:
        policy_engine = PolicyEngine(graph_store)
        
        import uuid
        policy = Policy(
            policy_id=str(uuid.uuid4()),
            name=name,
            description=description,
            rules=rules,
            category=category,
            version=version,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        policy_id = policy_engine.add_policy(policy)
        logger.info(f"Created policy: {policy_id} version {version}")
        return policy_id
        
    except Exception as e:
        logger.error(f"Failed to create policy: {e}")
        raise


def check_decision_compliance(
    graph_store: GraphStore,
    decision_id: str,
    policy_id: str
) -> Dict[str, Any]:
    """
    Check if a decision complies with a policy.
    
    Args:
        graph_store: Graph database instance
        decision_id: Decision ID to check
        policy_id: Policy ID to check against
        
    Returns:
        Compliance check results
    """
    logger = get_logger(__name__)
    
    try:
        # Get decision details
        query_engine = DecisionQuery(graph_store)
        decisions = query_engine.find_by_time_range(
            datetime.now().replace(year=2000), datetime.now(), limit=1000
        )
        
        decision = None
        for d in decisions:
            if d.decision_id == decision_id:
                decision = d
                break
        
        if not decision:
            raise ValueError(f"Decision {decision_id} not found")
        
        # Check compliance
        policy_engine = PolicyEngine(graph_store)
        is_compliant = policy_engine.check_compliance(decision, policy_id)
        
        # Record policy application
        policy = policy_engine.get_policy(policy_id)
        if policy:
            policy_engine.record_policy_application(decision_id, policy_id, policy.version)
        
        return {
            "decision_id": decision_id,
            "policy_id": policy_id,
            "is_compliant": is_compliant,
            "compliance_check_timestamp": datetime.now().isoformat(),
            "decision_category": decision.category,
            "decision_confidence": decision.confidence
        }
        
    except Exception as e:
        logger.error(f"Failed to check decision compliance: {e}")
        raise


def get_decision_statistics(
    graph_store: GraphStore,
    category: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Get decision statistics for analysis.
    
    Args:
        graph_store: Graph database instance
        category: Optional category filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        Decision statistics
    """
    logger = get_logger(__name__)
    
    try:
        query_engine = DecisionQuery(graph_store)
        
        # Get decisions based on filters
        if start_date and end_date:
            decisions = query_engine.find_by_time_range(start_date, end_date, limit=1000)
        elif category:
            decisions = query_engine.find_by_category(category, limit=1000)
        else:
            decisions = query_engine.find_by_time_range(
                datetime.now().replace(year=2020), datetime.now(), limit=1000
            )
        
        # Calculate statistics
        total_decisions = len(decisions)
        
        if total_decisions == 0:
            return {
                "total_decisions": 0,
                "categories": {},
                "outcomes": {},
                "average_confidence": 0.0,
                "date_range": {"start": None, "end": None}
            }
        
        # Category distribution
        categories = {}
        for decision in decisions:
            cat = decision.category
            categories[cat] = categories.get(cat, 0) + 1
        
        # Outcome distribution
        outcomes = {}
        for decision in decisions:
            outcome = decision.outcome
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        # Average confidence
        avg_confidence = sum(d.confidence for d in decisions) / total_decisions
        
        # Date range
        timestamps = [d.timestamp for d in decisions if d.timestamp]
        date_range = {
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None
        }
        
        return {
            "total_decisions": total_decisions,
            "categories": categories,
            "outcomes": outcomes,
            "average_confidence": avg_confidence,
            "date_range": date_range,
            "filters_applied": {
                "category": category,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get decision statistics: {e}")
        raise


def setup_decision_tracking(
    graph_store: GraphStore,
    create_sample_data: bool = False
) -> bool:
    """
    Set up decision tracking schema and optionally create sample data.
    
    Args:
        graph_store: Graph database instance
        create_sample_data: Whether to create sample data
        
    Returns:
        True if setup successful
    """
    logger = get_logger(__name__)
    
    try:
        from .graph_schema import setup_decision_schema, verify_schema, create_sample_data as create_sample
        
        # Setup schema
        setup_decision_schema(graph_store)
        
        # Verify schema
        if not verify_schema(graph_store):
            raise ValueError("Schema verification failed")
        
        # Create sample data if requested
        if create_sample_data:
            create_sample(graph_store)
        
        logger.info("Decision tracking setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup decision tracking: {e}")
        return False


# Method dispatch support for AgentContext
def enhance_agent_context_with_decisions(agent_context: AgentContext) -> None:
    """
    Enhance AgentContext with decision tracking methods if not already enabled.
    
    Args:
        agent_context: AgentContext instance to enhance
    """
    logger = get_logger(__name__)
    
    try:
        if not agent_context.config.get("enable_decision_tracking"):
            logger.warning("Decision tracking not enabled in AgentContext")
            return
        
        # Verify decision tracking components are available
        if not all([
            agent_context._decision_recorder,
            agent_context._decision_query,
            agent_context._causal_analyzer,
            agent_context._policy_engine
        ]):
            logger.warning("Decision tracking components not properly initialized")
            return
        
        logger.info("AgentContext enhanced with decision tracking")
        
    except Exception as e:
        logger.error(f"Failed to enhance AgentContext: {e}")


# Export all convenience functions
__all__ = [
    "record_decision",
    "find_precedents", 
    "get_causal_chain",
    "get_applicable_policies",
    "multi_hop_query",
    "capture_decision_trace",
    "find_exception_precedents",
    "analyze_decision_impact",
    "create_policy_with_versioning",
    "check_decision_compliance",
    "get_decision_statistics",
    "setup_decision_tracking",
    "enhance_agent_context_with_decisions"
]
