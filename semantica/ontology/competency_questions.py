"""
Competency Questions Manager

Manages competency questions that define what an ontology should answer,
serving as functional requirements that guide modeling decisions.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


@dataclass
class CompetencyQuestion:
    """Competency question definition."""
    question: str
    category: str = "general"
    priority: int = 1  # 1=high, 2=medium, 3=low
    answerable: bool = False
    trace_to_elements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompetencyQuestionsManager:
    """
    Competency questions manager for ontology requirements.
    
    • Define and manage competency questions
    • Validate ontology against competency questions
    • Trace questions to ontology elements
    • Refine questions based on ontology evolution
    • Generate question-answer validation reports
    • Support natural language question formulation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize competency questions manager.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("competency_questions_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.questions: List[CompetencyQuestion] = []
    
    def add_question(
        self,
        question: str,
        category: str = "general",
        priority: int = 1,
        **metadata
    ) -> CompetencyQuestion:
        """
        Add a competency question.
        
        Args:
            question: Question text
            category: Question category
            priority: Priority (1=high, 2=medium, 3=low)
            **metadata: Additional metadata
        
        Returns:
            Created competency question
        """
        cq = CompetencyQuestion(
            question=question,
            category=category,
            priority=priority,
            metadata=metadata
        )
        
        self.questions.append(cq)
        self.logger.info(f"Added competency question: {question[:50]}...")
        
        return cq
    
    def validate_ontology(
        self,
        ontology: Dict[str, Any],
        **options
    ) -> Dict[str, Any]:
        """
        Validate ontology against competency questions.
        
        Args:
            ontology: Ontology dictionary
            **options: Additional options
        
        Returns:
            Validation results
        """
        results = {
            "total_questions": len(self.questions),
            "answerable": 0,
            "unanswerable": 0,
            "by_category": {},
            "by_priority": {}
        }
        
        for question in self.questions:
            # Basic check if ontology can answer the question
            answerable = self._can_ontology_answer(ontology, question)
            question.answerable = answerable
            
            if answerable:
                results["answerable"] += 1
            else:
                results["unanswerable"] += 1
            
            # Track by category
            category = question.category
            if category not in results["by_category"]:
                results["by_category"][category] = {"answerable": 0, "unanswerable": 0}
            
            if answerable:
                results["by_category"][category]["answerable"] += 1
            else:
                results["by_category"][category]["unanswerable"] += 1
        
        return results
    
    def _can_ontology_answer(self, ontology: Dict[str, Any], question: CompetencyQuestion) -> bool:
        """Check if ontology can answer the question (basic heuristic)."""
        # Extract keywords from question
        question_lower = question.question.lower()
        
        # Check if ontology has relevant classes
        classes = ontology.get("classes", [])
        for cls in classes:
            class_name_lower = cls.get("name", "").lower()
            if any(word in class_name_lower for word in question_lower.split() if len(word) > 3):
                return True
        
        # Check if ontology has relevant properties
        properties = ontology.get("properties", [])
        for prop in properties:
            prop_name_lower = prop.get("name", "").lower()
            if any(word in prop_name_lower for word in question_lower.split() if len(word) > 3):
                return True
        
        return False
    
    def trace_to_elements(
        self,
        question: CompetencyQuestion,
        ontology: Dict[str, Any]
    ) -> List[str]:
        """
        Trace question to ontology elements.
        
        Args:
            question: Competency question
            ontology: Ontology dictionary
        
        Returns:
            List of relevant ontology element names
        """
        elements = []
        question_lower = question.question.lower()
        keywords = [w for w in question_lower.split() if len(w) > 3]
        
        # Find relevant classes
        classes = ontology.get("classes", [])
        for cls in classes:
            class_name_lower = cls.get("name", "").lower()
            if any(keyword in class_name_lower for keyword in keywords):
                elements.append(cls["name"])
        
        # Find relevant properties
        properties = ontology.get("properties", [])
        for prop in properties:
            prop_name_lower = prop.get("name", "").lower()
            if any(keyword in prop_name_lower for keyword in keywords):
                elements.append(prop["name"])
        
        question.trace_to_elements = elements
        return elements
    
    def generate_report(self, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate competency question validation report.
        
        Args:
            ontology: Ontology dictionary
        
        Returns:
            Validation report
        """
        validation = self.validate_ontology(ontology)
        
        # Trace all questions
        for question in self.questions:
            self.trace_to_elements(question, ontology)
        
        return {
            "validation": validation,
            "questions": [
                {
                    "question": q.question,
                    "category": q.category,
                    "priority": q.priority,
                    "answerable": q.answerable,
                    "trace_to_elements": q.trace_to_elements
                }
                for q in self.questions
            ],
            "generated_at": datetime.now().isoformat()
        }
    
    def get_questions_by_category(self, category: str) -> List[CompetencyQuestion]:
        """Get questions by category."""
        return [q for q in self.questions if q.category == category]
    
    def get_questions_by_priority(self, priority: int) -> List[CompetencyQuestion]:
        """Get questions by priority."""
        return [q for q in self.questions if q.priority == priority]
