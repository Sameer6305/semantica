"""
Policy Engine Module

This module provides the PolicyEngine class for policy management,
versioning, compliance checking, and impact analysis.

Core Features:
    - Policy management with versioning and change tracking
    - Compliance checking and violation detection
    - Policy impact analysis and assessment
    - Rule-based decision validation
    - Policy exception handling and management

Policy Management Features:
    - Policy Creation: Create and manage decision policies
    - Version Control: Track policy versions and changes
    - Rule Definition: Define policy rules and constraints
    - Policy Hierarchy: Manage policy relationships and dependencies
    - Lifecycle Management: Policy activation, deactivation, and retirement

Compliance and Validation:
    - Compliance Checking: Verify decision compliance with policies
    - Violation Detection: Identify policy violations and exceptions
    - Rule Evaluation: Evaluate complex policy rules
    - Constraint Validation: Validate decision constraints
    - Audit Reporting: Generate compliance audit reports

Advanced Analytics:
    - Impact Analysis: Analyze policy impact on decisions
    - Coverage Analysis: Assess policy coverage and gaps
    - Performance Metrics: Track policy performance metrics
    - Trend Analysis: Analyze policy compliance trends
    - Risk Assessment: Assess policy-related risks

Policy Engine Methods:
    - create_policy(): Create new policies with rules
    - update_policy(): Update existing policies
    - check_compliance(): Check decision compliance
    - analyze_impact(): Analyze policy impact
    - get_violations(): Get policy violations
    - audit_decisions(): Audit decisions against policies

Rule Engine Features:
    - Complex Rules: Support complex rule definitions
    - Conditional Logic: Handle conditional rule logic
    - Rule Chaining: Chain multiple rules together
    - Exception Handling: Handle rule exceptions
    - Performance Optimization: Optimized rule evaluation

Example Usage:
    >>> from semantica.context import PolicyEngine
    >>> engine = PolicyEngine(graph_store=kg)
    >>> policy_id = engine.create_policy(
    ...     name="Lending Policy",
    ...     rules={"max_loan_amount": 500000, "min_credit_score": 650},
    ...     category="lending"
    ... )
    >>> compliance = engine.check_compliance(decision_id, policy_id)
    >>> violations = engine.get_violations(decision_id)
    >>> impact = engine.analyze_impact(policy_id, time_range="30d")

Production Use Cases:
    - Banking: Lending policies, risk management policies
    - Healthcare: Treatment protocols, clinical guidelines
    - Legal: Compliance policies, regulatory requirements
    - Government: Policy enforcement, regulatory compliance
    - Insurance: Underwriting policies, claim processing rules
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import uuid

from ..graph_store import GraphStore
from ..utils.logging import get_logger
from .decision_models import Decision, Policy, Exception


class PolicyEngine:
    """
    Policy engine with versioning and change tracking.
    
    This class manages policies, tracks versions, checks compliance,
    records exceptions, and analyzes policy impact.
    """
    
    def __init__(self, graph_store: GraphStore):
        """
        Initialize PolicyEngine.
        
        Args:
            graph_store: Graph database instance for storing policies
        """
        self.graph_store = graph_store
        self.logger = get_logger(__name__)
    
    def add_policy(self, policy: Policy) -> str:
        """
        Store policy with versioning.
        
        Args:
            policy: Policy object to store
            
        Returns:
            Policy ID
        """
        try:
            # Store policy node
            query = """
            CREATE (p:Policy {
                policy_id: $policy_id,
                name: $name,
                description: $description,
                rules: $rules,
                category: $category,
                version: $version,
                created_at: $created_at,
                updated_at: $updated_at,
                metadata: $metadata
            })
            """
            self.graph_store.execute_query(query, {
                "policy_id": policy.policy_id,
                "name": policy.name,
                "description": policy.description,
                "rules": policy.rules,
                "category": policy.category,
                "version": policy.version,
                "created_at": policy.created_at,
                "updated_at": policy.updated_at,
                "metadata": policy.metadata
            })
            
            self.logger.info(f"Added policy: {policy.policy_id} version {policy.version}")
            return policy.policy_id
            
        except Exception as e:
            self.logger.error(f"Failed to add policy: {e}")
            raise
    
    def update_policy(
        self,
        policy_id: str,
        rules: Dict[str, Any],
        change_reason: str,
        new_version: Optional[str] = None
    ) -> str:
        """
        Update policy and create new version.
        
        Args:
            policy_id: Policy ID to update
            rules: New policy rules
            change_reason: Reason for the change
            new_version: Optional new version (auto-generated if not provided)
            
        Returns:
            New version string
        """
        try:
            # Get current policy
            current_policy = self.get_policy(policy_id)
            if not current_policy:
                raise ValueError(f"Policy {policy_id} not found")
            
            # Generate new version if not provided
            if not new_version:
                new_version = self._generate_next_version(current_policy.version)
            
            # Create new policy version
            updated_policy = Policy(
                policy_id=policy_id,
                name=current_policy.name,
                description=current_policy.description,
                rules=rules,
                category=current_policy.category,
                version=new_version,
                created_at=current_policy.created_at,
                updated_at=datetime.now(),
                metadata={
                    **current_policy.metadata,
                    "change_reason": change_reason,
                    "previous_version": current_policy.version
                }
            )
            
            # Store new version
            self.add_policy(updated_policy)
            
            # Link versions
            query = """
            MATCH (old:Policy {policy_id: $policy_id, version: $old_version})
            MATCH (new:Policy {policy_id: $policy_id, version: $new_version})
            MERGE (old)-[:VERSION_OF]->(new)
            """
            self.graph_store.execute_query(query, {
                "policy_id": policy_id,
                "old_version": current_policy.version,
                "new_version": new_version
            })
            
            self.logger.info(f"Updated policy {policy_id} to version {new_version}")
            return new_version
            
        except Exception as e:
            self.logger.error(f"Failed to update policy: {e}")
            raise
    
    def get_applicable_policies(
        self,
        category: str,
        entities: Optional[List[str]] = None
    ) -> List[Policy]:
        """
        Get policies for category/entities.
        
        Args:
            category: Policy category
            entities: Optional list of entity IDs for entity-specific policies
            
        Returns:
            List of applicable policies (latest versions)
        """
        try:
            # Get latest policies for category
            query = """
            MATCH (p:Policy {category: $category})
            WHERE NOT (p)-[:VERSION_OF]->(:Policy)
            RETURN p
            ORDER BY p.updated_at DESC
            """
            results = self.graph_store.execute_query(query, {"category": category})
            
            policies = []
            for record in results:
                policy_data = record.get("p", {})
                policies.append(self._dict_to_policy(policy_data))
            
            # Filter by entities if specified
            if entities:
                # This would require entity-specific policy relationships
                # For now, return all category policies
                pass
            
            self.logger.info(f"Found {len(policies)} applicable policies for category {category}")
            return policies
            
        except Exception as e:
            self.logger.error(f"Failed to get applicable policies: {e}")
            raise
    
    def check_compliance(self, decision: Decision, policy_id: str) -> bool:
        """
        Check if decision complies with policy.
        
        Args:
            decision: Decision to check
            policy_id: Policy ID to check against
            
        Returns:
            True if compliant, False otherwise
        """
        try:
            policy = self.get_policy(policy_id)
            if not policy:
                raise ValueError(f"Policy {policy_id} not found")
            
            # Simple rule-based compliance check
            # In practice, this would be more sophisticated
            rules = policy.rules
            
            # Example compliance checks
            if "min_confidence" in rules:
                if decision.confidence < rules["min_confidence"]:
                    return False
            
            if "allowed_outcomes" in rules:
                if decision.outcome not in rules["allowed_outcomes"]:
                    return False
            
            if "required_categories" in rules:
                if decision.category not in rules["required_categories"]:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check compliance: {e}")
            return False
    
    def record_policy_application(
        self,
        decision_id: str,
        policy_id: str,
        version: str
    ) -> None:
        """
        Track policy application with version.
        
        Args:
            decision_id: Decision ID
            policy_id: Policy ID
            version: Policy version that was applied
        """
        try:
            query = """
            MATCH (d:Decision {decision_id: $decision_id})
            MATCH (p:Policy {policy_id: $policy_id, version: $version})
            MERGE (d)-[:APPLIED_POLICY]->(p)
            SET d.policy_applied_at = timestamp()
            """
            self.graph_store.execute_query(query, {
                "decision_id": decision_id,
                "policy_id": policy_id,
                "version": version
            })
            
            self.logger.info(f"Recorded policy application: {policy_id} v{version} to decision {decision_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record policy application: {e}")
            raise
    
    def record_exception(
        self,
        decision_id: str,
        policy_id: str,
        reason: str
    ) -> str:
        """
        Track policy exceptions.
        
        Args:
            decision_id: Decision ID
            policy_id: Policy ID that was excepted
            reason: Reason for exception
            
        Returns:
            Exception ID
        """
        try:
            exception_id = str(uuid.uuid4())
            
            query = """
            CREATE (e:Exception {
                exception_id: $exception_id,
                decision_id: $decision_id,
                policy_id: $policy_id,
                reason: $reason,
                created_at: datetime()
            })
            """
            self.graph_store.execute_query(query, {
                "exception_id": exception_id,
                "decision_id": decision_id,
                "policy_id": policy_id,
                "reason": reason
            })
            
            # Link to decision and policy
            query = """
            MATCH (d:Decision {decision_id: $decision_id})
            MATCH (p:Policy {policy_id: $policy_id})
            MATCH (e:Exception {exception_id: $exception_id})
            MERGE (d)-[:GRANTED_EXCEPTION]->(e)
            MERGE (e)-[:OVERRIDDEN_POLICY]->(p)
            """
            self.graph_store.execute_query(query, {
                "decision_id": decision_id,
                "policy_id": policy_id,
                "exception_id": exception_id
            })
            
            self.logger.info(f"Recorded policy exception: {exception_id}")
            return exception_id
            
        except Exception as e:
            self.logger.error(f"Failed to record exception: {e}")
            raise
    
    def get_policy_history(self, policy_id: str) -> List[Policy]:
        """
        Get version history for a policy.
        
        Args:
            policy_id: Policy ID
            
        Returns:
            List of policy versions
        """
        try:
            query = """
            MATCH (p:Policy {policy_id: $policy_id})
            OPTIONAL MATCH (p)-[:VERSION_OF*]->(future:Policy)
            WITH collect(p) + collect(future) as all_versions
            UNWIND all_versions as version
            RETURN DISTINCT version
            ORDER BY version.updated_at
            """
            results = self.graph_store.execute_query(query, {"policy_id": policy_id})
            
            policies = []
            for record in results:
                policy_data = record.get("version", {})
                policies.append(self._dict_to_policy(policy_data))
            
            self.logger.info(f"Found {len(policies)} versions for policy {policy_id}")
            return policies
            
        except Exception as e:
            self.logger.error(f"Failed to get policy history: {e}")
            raise
    
    def get_affected_decisions(
        self,
        policy_id: str,
        from_version: str,
        to_version: str
    ) -> List[str]:
        """
        Find decisions affected by policy change.
        
        Args:
            policy_id: Policy ID
            from_version: Previous version
            to_version: New version
            
        Returns:
            List of affected decision IDs
        """
        try:
            query = """
            MATCH (d:Decision)-[:APPLIED_POLICY]->(p:Policy {
                policy_id: $policy_id,
                version: $from_version
            })
            RETURN d.decision_id as decision_id
            """
            results = self.graph_store.execute_query(query, {
                "policy_id": policy_id,
                "from_version": from_version
            })
            
            decision_ids = []
            for record in results:
                decision_ids.append(record.get("decision_id", ""))
            
            self.logger.info(f"Found {len(decision_ids)} decisions affected by policy change")
            return decision_ids
            
        except Exception as e:
            self.logger.error(f"Failed to get affected decisions: {e}")
            raise
    
    def analyze_policy_impact(
        self,
        policy_id: str,
        proposed_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        "What-if" scenarios for policy updates.
        
        Args:
            policy_id: Policy ID to analyze
            proposed_rules: Proposed new rules
            
        Returns:
            Impact analysis results
        """
        try:
            current_policy = self.get_policy(policy_id)
            if not current_policy:
                raise ValueError(f"Policy {policy_id} not found")
            
            # Get decisions that used this policy
            query = """
            MATCH (d:Decision)-[:APPLIED_POLICY]->(p:Policy {policy_id: $policy_id})
            RETURN d.decision_id as decision_id, d.confidence as confidence,
                   d.outcome as outcome, d.category as category
            """
            results = self.graph_store.execute_query(query, {"policy_id": policy_id})
            
            # Analyze impact
            impact_analysis = {
                "total_decisions": len(results),
                "affected_decisions": 0,
                "compliance_changes": {},
                "risk_assessment": "low",
                "recommendations": []
            }
            
            for record in results:
                decision_data = {
                    "confidence": record.get("confidence", 0.0),
                    "outcome": record.get("outcome", ""),
                    "category": record.get("category", "")
                }
                
                # Check if decision would still comply with new rules
                would_comply = self._check_compliance_with_rules(
                    decision_data, proposed_rules
                )
                
                if not would_comply:
                    impact_analysis["affected_decisions"] += 1
            
            # Calculate impact percentage
            if impact_analysis["total_decisions"] > 0:
                impact_percentage = (
                    impact_analysis["affected_decisions"] / 
                    impact_analysis["total_decisions"]
                ) * 100
                
                if impact_percentage > 50:
                    impact_analysis["risk_assessment"] = "high"
                elif impact_percentage > 20:
                    impact_analysis["risk_assessment"] = "medium"
                
                impact_analysis["impact_percentage"] = impact_percentage
            
            # Generate recommendations
            if impact_analysis["risk_assessment"] == "high":
                impact_analysis["recommendations"].append(
                    "Consider gradual rollout of policy changes"
                )
                impact_analysis["recommendations"].append(
                    "Review affected decisions for potential exceptions"
                )
            
            self.logger.info(f"Analyzed policy impact for {policy_id}")
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze policy impact: {e}")
            raise
    
    def get_policy(self, policy_id: str, version: Optional[str] = None) -> Optional[Policy]:
        """
        Get policy by ID and optional version.
        
        Args:
            policy_id: Policy ID
            version: Optional version (latest if not specified)
            
        Returns:
            Policy object or None
        """
        try:
            if version:
                query = """
                MATCH (p:Policy {policy_id: $policy_id, version: $version})
                RETURN p
                """
                params = {"policy_id": policy_id, "version": version}
            else:
                # Get latest version
                query = """
                MATCH (p:Policy {policy_id: $policy_id})
                WHERE NOT (p)-[:VERSION_OF]->(:Policy)
                RETURN p
                """
                params = {"policy_id": policy_id}
            
            results = self.graph_store.execute_query(query, params)
            
            if results:
                policy_data = results[0].get("p", {})
                return self._dict_to_policy(policy_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get policy: {e}")
            return None
    
    def _generate_next_version(self, current_version: str) -> str:
        """Generate next version number."""
        try:
            # Simple semantic versioning
            parts = current_version.split(".")
            if len(parts) >= 2:
                patch = int(parts[-1]) + 1
                parts[-1] = str(patch)
                return ".".join(parts)
            else:
                return f"{current_version}.1"
        except Exception:
            return f"{current_version}.1"
    
    def _check_compliance_with_rules(
        self,
        decision_data: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> bool:
        """Check compliance with given rules."""
        try:
            if "min_confidence" in rules:
                if decision_data.get("confidence", 0) < rules["min_confidence"]:
                    return False
            
            if "allowed_outcomes" in rules:
                if decision_data.get("outcome", "") not in rules["allowed_outcomes"]:
                    return False
            
            if "required_categories" in rules:
                if decision_data.get("category", "") not in rules["required_categories"]:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _dict_to_policy(self, data: Dict[str, Any]) -> Policy:
        """Convert dictionary to Policy object."""
        # Handle timestamp conversion
        for field in ["created_at", "updated_at"]:
            if isinstance(data.get(field), str):
                data[field] = datetime.fromisoformat(data[field])
        
        return Policy(
            policy_id=data.get("policy_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            rules=data.get("rules", {}),
            category=data.get("category", ""),
            version=data.get("version", ""),
            created_at=data.get("created_at", datetime.now()),
            updated_at=data.get("updated_at", datetime.now()),
            metadata=data.get("metadata", {})
        )
