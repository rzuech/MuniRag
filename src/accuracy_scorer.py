"""
Multi-dimensional accuracy scoring system for MuniRAG
Evaluates LLM responses against expected answers
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
from src.logger import get_logger

logger = get_logger("accuracy_scorer")


class AccuracyScorer:
    """Scores LLM responses using multi-dimensional criteria"""
    
    def __init__(self, rubric_path: str = "test_questions.json"):
        """Initialize scorer with test questions and rubric"""
        with open(rubric_path, 'r') as f:
            self.test_data = json.load(f)
        
        self.rubric = self.test_data["scoring_rubric"]
        self.thresholds = self.test_data["grade_thresholds"]
        
    def score_response(self, 
                      response: str, 
                      question_data: Dict,
                      sources: Optional[List[str]] = None) -> Dict:
        """
        Score a single response against expected answer
        
        Args:
            response: LLM's response text
            question_data: Question config with expected elements
            sources: List of source citations
            
        Returns:
            Score breakdown with overall score and details
        """
        # Normalize response for analysis
        response_lower = response.lower()
        response_tokens = set(re.findall(r'\b\w+\b', response_lower))
        
        # Calculate sub-scores
        factual_score = self._score_factual_accuracy(response_lower, response_tokens, question_data)
        completeness_score = self._score_completeness(response_lower, response_tokens, question_data)
        relevance_score = self._score_relevance(response, question_data, sources)
        coherence_score = self._score_coherence(response)
        
        # Calculate weighted overall score
        overall_score = (
            factual_score * self.rubric["factual_accuracy"]["weight"] +
            completeness_score * self.rubric["completeness"]["weight"] +
            relevance_score * self.rubric["relevance"]["weight"] +
            coherence_score * self.rubric["coherence"]["weight"]
        )
        
        # Determine grade
        grade = self._get_grade(overall_score)
        
        return {
            "overall_score": round(overall_score, 3),
            "grade": grade,
            "sub_scores": {
                "factual_accuracy": round(factual_score, 3),
                "completeness": round(completeness_score, 3),
                "relevance": round(relevance_score, 3),
                "coherence": round(coherence_score, 3)
            },
            "details": {
                "response_length": len(response),
                "sources_cited": len(sources) if sources else 0,
                "required_elements_found": self._check_required_elements(response_lower, question_data),
                "bonus_elements_found": self._check_bonus_elements(response_lower, question_data),
                "penalties_triggered": self._check_penalties(response_lower, question_data)
            }
        }
    
    def _score_factual_accuracy(self, response_lower: str, response_tokens: set, question_data: Dict) -> float:
        """Score factual accuracy based on required and bonus elements"""
        criteria = self.rubric["factual_accuracy"]["criteria"]
        score = 0.0
        
        # Check required elements
        required = question_data.get("required_elements", [])
        if required:
            found_required = sum(1 for elem in required if elem.lower() in response_lower)
            required_ratio = found_required / len(required)
            score += required_ratio * criteria["all_required_elements"]
        else:
            # If no required elements, give full credit for this portion
            score += criteria["all_required_elements"]
        
        # Check bonus elements
        bonus = question_data.get("bonus_elements", [])
        if bonus:
            found_bonus = sum(1 for elem in bonus if elem.lower() in response_lower)
            bonus_ratio = found_bonus / len(bonus)
            score += bonus_ratio * criteria["bonus_elements"]
        else:
            score += criteria["bonus_elements"]
        
        # Check for wrong information (penalties)
        penalties = question_data.get("wrong_answer_penalties", [])
        if penalties:
            penalty_count = sum(1 for penalty in penalties if penalty.lower() in response_lower)
            if penalty_count == 0:
                score += criteria["no_wrong_information"]
            else:
                # Partial credit if only minor penalties
                score += criteria["no_wrong_information"] * max(0, 1 - (penalty_count / len(penalties)))
        else:
            score += criteria["no_wrong_information"]
        
        return min(1.0, score)  # Cap at 1.0
    
    def _score_completeness(self, response_lower: str, response_tokens: set, question_data: Dict) -> float:
        """Score how completely the response addresses the question"""
        criteria = self.rubric["completeness"]["criteria"]
        score = 0.0
        
        # Check if response addresses the question
        question_words = set(re.findall(r'\b\w+\b', question_data["question"].lower()))
        question_words -= {'what', 'how', 'when', 'where', 'why', 'is', 'are', 'the', 'a', 'an'}
        
        overlap = len(question_words & response_tokens)
        if overlap > 0:
            score += criteria["addresses_question"]
        
        # Check sufficient detail (response length as proxy)
        if len(response_lower) > 100:  # Reasonable detail
            score += criteria["sufficient_detail"]
        elif len(response_lower) > 50:  # Some detail
            score += criteria["sufficient_detail"] * 0.5
        
        # Check for context inclusion
        if any(word in response_lower for word in ['because', 'therefore', 'however', 'additionally']):
            score += criteria["includes_context"]
        
        return min(1.0, score)
    
    def _score_relevance(self, response: str, question_data: Dict, sources: Optional[List[str]]) -> float:
        """Score relevance and source attribution"""
        criteria = self.rubric["relevance"]["criteria"]
        score = 0.0
        
        # On-topic check (simple heuristic)
        category_keywords = {
            "factual": ["is", "are", "means", "defined"],
            "process": ["step", "how", "procedure", "first", "then", "next"],
            "definitional": ["definition", "means", "refers to", "is a"],
            "contextual": ["depends", "if", "when", "generally", "typically"]
        }
        
        category = question_data.get("category", "factual")
        if any(kw in response.lower() for kw in category_keywords.get(category, [])):
            score += criteria["on_topic"]
        
        # Source citation check
        if sources and len(sources) > 0:
            score += criteria["cites_source"]
        elif any(indicator in response.lower() for indicator in ["according to", "states that", "section", "article"]):
            score += criteria["cites_source"] * 0.5  # Partial credit for implicit citation
        
        # Minimal irrelevant info (shorter responses get higher scores here)
        if len(response) < 500:
            score += criteria["minimal_irrelevant"]
        elif len(response) < 1000:
            score += criteria["minimal_irrelevant"] * 0.7
        else:
            score += criteria["minimal_irrelevant"] * 0.4
        
        return min(1.0, score)
    
    def _score_coherence(self, response: str) -> float:
        """Score logical flow and clarity"""
        criteria = self.rubric["coherence"]["criteria"]
        score = 0.0
        
        # Check logical flow (presence of transition words)
        transitions = ['first', 'second', 'then', 'next', 'finally', 'however', 
                      'therefore', 'additionally', 'furthermore', 'moreover']
        if any(trans in response.lower() for trans in transitions):
            score += criteria["logical_flow"]
        
        # Clear language (no excessive jargon, reasonable sentence length)
        sentences = response.split('.')
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_sentence_length = np.mean(sentence_lengths)
            if 10 <= avg_sentence_length <= 25:  # Reasonable sentence length
                score += criteria["clear_language"]
        else:
            # If no sentences found, give partial credit
            score += criteria["clear_language"] * 0.5
        
        # No contradictions (basic check - would need NLP for thorough check)
        # For now, give full credit unless obvious issues
        score += criteria["no_contradictions"]
        
        return min(1.0, score)
    
    def _check_required_elements(self, response_lower: str, question_data: Dict) -> List[str]:
        """Return list of found required elements"""
        required = question_data.get("required_elements", [])
        return [elem for elem in required if elem.lower() in response_lower]
    
    def _check_bonus_elements(self, response_lower: str, question_data: Dict) -> List[str]:
        """Return list of found bonus elements"""
        bonus = question_data.get("bonus_elements", [])
        return [elem for elem in bonus if elem.lower() in response_lower]
    
    def _check_penalties(self, response_lower: str, question_data: Dict) -> List[str]:
        """Return list of triggered penalties"""
        penalties = question_data.get("wrong_answer_penalties", [])
        return [penalty for penalty in penalties if penalty.lower() in response_lower]
    
    def _get_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        for grade, threshold in sorted(self.thresholds.items(), key=lambda x: -x[1]):
            if score >= threshold:
                return grade
        return "failing"
    
    def score_test_suite(self, results: List[Dict]) -> Dict:
        """
        Score an entire test suite run
        
        Args:
            results: List of test results with responses
            
        Returns:
            Aggregate scores and statistics
        """
        total_score = 0.0
        category_scores = {}
        difficulty_scores = {}
        pdf_scores = {}
        
        for result in results:
            score_data = result["score_data"]
            question_meta = result["question_metadata"]
            
            # Aggregate total
            total_score += score_data["overall_score"]
            
            # By category
            category = question_meta.get("category", "unknown")
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score_data["overall_score"])
            
            # By difficulty
            difficulty = question_meta.get("difficulty", "unknown")
            if difficulty not in difficulty_scores:
                difficulty_scores[difficulty] = []
            difficulty_scores[difficulty].append(score_data["overall_score"])
            
            # By PDF
            pdf = question_meta.get("pdf", "unknown")
            if pdf not in pdf_scores:
                pdf_scores[pdf] = []
            pdf_scores[pdf].append(score_data["overall_score"])
        
        # Calculate averages
        avg_score = total_score / len(results) if results else 0
        
        return {
            "overall_average": round(avg_score, 3),
            "overall_grade": self._get_grade(avg_score),
            "total_questions": len(results),
            "category_averages": {k: round(np.mean(v), 3) for k, v in category_scores.items()},
            "difficulty_averages": {k: round(np.mean(v), 3) for k, v in difficulty_scores.items()},
            "pdf_averages": {k: round(np.mean(v), 3) for k, v in pdf_scores.items()},
            "summary": self._generate_summary(avg_score, category_scores, difficulty_scores)
        }
    
    def _generate_summary(self, avg_score: float, category_scores: Dict, difficulty_scores: Dict) -> str:
        """Generate human-readable summary of results"""
        grade = self._get_grade(avg_score)
        
        # Find weakest category
        weakest_category = min(category_scores.items(), key=lambda x: np.mean(x[1]))[0] if category_scores else "unknown"
        
        summary = f"Overall Performance: {grade.upper()} ({avg_score:.1%})\n"
        
        if grade == "excellent":
            summary += "The system is performing exceptionally well across all question types."
        elif grade == "good":
            summary += f"Good performance overall. Consider improving {weakest_category} questions."
        elif grade == "acceptable":
            summary += f"Acceptable performance but significant room for improvement, especially in {weakest_category} questions."
        else:
            summary += f"Performance needs improvement. Focus on {weakest_category} questions and review retrieval settings."
        
        return summary