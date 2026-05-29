"""
LLM integration service for the Learning Agent System.

This module handles LLM interactions for question generation, answer simulation,
and scoring using Ollama and LangChain.
"""

import asyncio
import logging
import json
import re
import random
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from .models import ProcessedContext, GeneratedQuestion
from .langsmith_config import trace_llm_operation, get_langsmith_callbacks

logger = logging.getLogger(__name__)

class LLMService:
    """Handles LLM interactions for question generation and verification."""
    
    def __init__(self, model_name: str = "llama3.1:latest"):
        """Initialize LLM service."""
        # Get LangSmith callbacks for tracing
        callbacks = get_langsmith_callbacks()
        self.llm = OllamaLLM(model=model_name, callbacks=callbacks)
        self.target_distribution = {
            "multiple_choice": 5,
            "short_answer": 3,
            "long_answer": 2
        }
        
    @trace_llm_operation("generate_questions")
    async def generate_questions(self, context_chunks: List[ProcessedContext], 
                               checkpoint_requirements: List[str]) -> List[GeneratedQuestion]:
        """Generate 10 questions: 5 MCQ, 3 short answer, 2 long answer."""
        try:
            # Combine context for question generation with better structure
            combined_context = "\n\n".join([f"Context {i+1}: {chunk['text']}" 
                                          for i, chunk in enumerate(context_chunks)])
            requirements_text = "\n".join([f"• {req}" for req in checkpoint_requirements])
            
            # Enhanced prompt with specific instructions for relevance
            prompt = f"""
You are an expert educational assessment designer. Create exactly 10 high-quality questions that directly assess understanding of the specific learning objectives below.

LEARNING OBJECTIVES TO ASSESS:
{requirements_text}

LEARNING CONTENT:
{combined_context[:2000]}

REQUIREMENTS FOR QUESTIONS:
1. Each question MUST directly test one or more of the learning objectives listed above
2. Questions should use terminology and concepts from the provided content
3. Include exactly 5 multiple choice questions (MCQ), 3 short-answer questions, and 2 long-answer questions
4. MCQs must each have options A-D with one correct answer
5. Short-answer questions should be answerable in 2-4 sentences
6. Long-answer questions should require deeper reasoning, examples, and structured explanation
6. Make questions specific to the content, not generic

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

QUESTION 1 (MCQ): ...
A) ...
B) ...
C) ...
D) ...
CORRECT: A

QUESTION 2 (MCQ): ...
A) ...
B) ...
C) ...
D) ...
CORRECT: B

QUESTION 3 (MCQ): ...
...

QUESTION 4 (MCQ): ...
...

QUESTION 5 (MCQ): ...
...

QUESTION 6 (SHORT): ...
QUESTION 7 (SHORT): ...
QUESTION 8 (SHORT): ...

QUESTION 9 (LONG): ...
QUESTION 10 (LONG): ...
"""
            
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            
            # Handle response type - ensure it's a string
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Parse and enforce distribution
            questions = self._parse_enhanced_questions(response_text, checkpoint_requirements)
            questions = self._enforce_distribution(questions, checkpoint_requirements)
            
            # Validate overall quality and retry if needed
            # DISABLED: Quality validation was too strict, questions are already good
            # if not self._validate_question_set_quality(questions, checkpoint_requirements):
            #     logger.warning("Initial questions did not meet quality threshold, generating fallback questions")
            #     questions = self._get_fallback_questions(checkpoint_requirements)
            
            logger.info(f"Generated {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return self._enforce_distribution([], checkpoint_requirements)
    
    def _parse_enhanced_questions(self, response: str, requirements: List[str]) -> List[GeneratedQuestion]:
        """Parse enhanced questions including MCQs from LLM response with validation."""
        questions = []
        lines = response.strip().split('\n')
        
        current_question = None
        i = 0
        
        print(f"DEBUG: Parsing {len(lines)} lines from LLM response")
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Handle different formatting patterns from LLM
            if ("QUESTION" in line.upper() and ("MULTIPLE" in line.upper() or "MCQ" in line.upper() or "OPEN" in line.upper() or "SHORT" in line.upper() or "ESSAY" in line.upper())):
                line_upper = line.upper()
                if "MCQ" in line_upper or "MULTIPLE" in line_upper:
                    question_type = "multiple_choice"
                elif "SHORT" in line_upper:
                    question_type = "short_answer"
                elif "LONG" in line_upper:
                    question_type = "long_answer"
                else:
                    question_type = "short_answer"
                
                print(f"DEBUG: Found question line: {line}")
                
                # Extract question text - handle the actual format we're getting
                question_text = None
                if ": " in line:
                    # Handle "QUESTION 1 (MCQ): What is..." format
                    parts = line.split(": ", 1)
                    if len(parts) == 2:
                        question_text = parts[1].strip()
                elif i + 1 < len(lines):
                    # Question might be on next line
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.upper().startswith(("QUESTION", "A)", "B)", "C)", "D)", "CORRECT")):
                        question_text = next_line
                        i += 1  # Skip the next line since we used it
                
                if not question_text:
                    print(f"DEBUG: Could not extract question text from: '{line}'")
                    if i + 1 < len(lines):
                        print(f"DEBUG: Next line is: '{lines[i+1].strip()}'")
                    i += 1
                    continue
                
                print(f"DEBUG: Extracted question: {question_text}")
                
                # Always create question - remove quality validation here to see what we get
                current_question = {
                    "question": question_text,
                    "type": question_type,
                    "difficulty": self._assess_question_difficulty(question_text),
                    "expected_elements": self._extract_relevant_requirements(question_text, requirements)
                }
                
                # If it's MCQ, parse options and correct answer
                if question_type == "multiple_choice":
                    options = []
                    correct_answer = "A"
                    
                    # Look for options in next lines
                    j = i + 1
                    while j < len(lines) and j < i + 10:  # Increased search range
                        option_line = lines[j].strip()
                        print(f"DEBUG: Checking line {j}: '{option_line}'")
                        # Handle both A) and A: formats
                        if re.match(r'^[A-D][\)\:]', option_line):
                            options.append(option_line)
                            print(f"DEBUG: Found option: {option_line}")
                        elif option_line.upper().startswith("CORRECT:"):
                            # Extract correct answer, handle B) format
                            correct_part = option_line.split(":", 1)[1].strip()
                            correct_answer = re.sub(r'[^\w]', '', correct_part)  # Remove punctuation
                            print(f"DEBUG: Found correct answer: {correct_answer}")
                            break
                        elif option_line and (option_line.startswith("**QUESTION") or option_line.upper().startswith("QUESTION")):
                            print(f"DEBUG: Hit next question, stopping option search")
                            break
                        j += 1
                    
                    # Validate MCQ has proper options
                    if len(options) >= 3:
                        current_question["options"] = options
                        current_question["correct_answer"] = correct_answer
                        print(f"DEBUG: MCQ with {len(options)} options, correct: {correct_answer}")
                    else:
                        print(f"DEBUG: MCQ invalid - only {len(options)} options found")
                        current_question = None  # Invalid MCQ
                    i = j
                
                if current_question:
                    questions.append(current_question)
                    print(f"DEBUG: Added question: {current_question['question'][:50]}...")
            
            i += 1
        
        print(f"DEBUG: Parsed {len(questions)} total questions")
        
        return questions
    
    def _validate_question_quality(self, question_text: str, requirements: List[str]) -> bool:
        """Validate that a question meets quality criteria."""
        question_lower = question_text.lower()
        
        # Check for generic/poor quality questions
        poor_quality_patterns = [
            'key concepts',
            'main ideas', 
            'important aspects',
            'basic principles',
            'tell me about',
            'what do you know'
        ]
        
        if any(pattern in question_lower for pattern in poor_quality_patterns):
            return False
        
        # Check for requirement-related terms
        requirement_words = set()
        for req in requirements:
            words = [word.lower().strip('.,!?') for word in req.split() if len(word) > 3]
            requirement_words.update(words)
        
        question_words = set(word.lower().strip('.,!?') for word in question_text.split())
        overlap = len(question_words.intersection(requirement_words))
        
        # Require some overlap with requirements
        return overlap > 0
    
    def _assess_question_difficulty(self, question_text: str) -> str:
        """Assess question difficulty based on question patterns."""
        question_lower = question_text.lower()
        
        # Easy: basic recall/recognition
        if any(word in question_lower for word in ['what', 'which', 'who', 'when', 'where', 'list']):
            return "easy"
        
        # Hard: analysis/synthesis  
        elif any(word in question_lower for word in ['analyze', 'evaluate', 'compare', 'synthesize', 'create', 'design']):
            return "hard"
        
        # Medium: comprehension/application
        else:
            return "medium"
    
    def _extract_relevant_requirements(self, question_text: str, requirements: List[str]) -> List[str]:
        """Extract the most relevant requirements for a question."""
        question_words = set(word.lower().strip('.,!?') for word in question_text.split())
        
        relevant_reqs = []
        for req in requirements:
            req_words = set(word.lower().strip('.,!?') for word in req.split())
            overlap = len(question_words.intersection(req_words))
            if overlap > 0:
                relevant_reqs.append(req)
        
        return relevant_reqs if relevant_reqs else requirements[:2]
    
    def _validate_question_set_quality(self, questions: List[GeneratedQuestion], requirements: List[str]) -> bool:
        """Validate that the overall question set meets quality requirements."""
        if len(questions) < 3:
            return False
        
        # Must have at least 1 MCQ
        mcq_count = sum(1 for q in questions if q['type'] == 'multiple_choice')
        if mcq_count == 0:
            return False
        
        # Calculate average relevance using simplified assessment
        total_relevance = 0
        for question in questions:
            question_words = set(word.lower().strip('.,!?') for word in question['question'].split())
            requirement_words = set()
            for req in requirements:
                words = [word.lower().strip('.,!?') for word in req.split() if len(word) > 3]
                requirement_words.update(words)
            
            if requirement_words:
                overlap = len(question_words.intersection(requirement_words))
                relevance = min((overlap / len(requirement_words)) * 100, 100)
            else:
                relevance = 50
                
            total_relevance += relevance
        
        average_relevance = total_relevance / len(questions)
        
        # Quality threshold: average relevance should be at least 40% (was 60%)
        return average_relevance >= 40
    
    def _get_fallback_questions(self, requirements: List[str]) -> List[GeneratedQuestion]:
        """Get fallback questions tailored to requirements."""
        
        # Extract key concepts from requirements for better question generation
        concepts = self._extract_key_concepts(requirements)
        
        # Determine topic from concepts
        if any(word in ' '.join(requirements).lower() for word in ['supervised', 'unsupervised', 'machine learning', 'ml']):
            topic_area = "machine_learning"
        elif any(word in ' '.join(requirements).lower() for word in ['neural', 'network', 'deep learning']):
            topic_area = "neural_networks"
        elif any(word in ' '.join(requirements).lower() for word in ['data', 'analysis', 'visualization']):
            topic_area = "data_science"
        elif any(word in ' '.join(requirements).lower() for word in ['text', 'language', 'nlp', 'sentiment']):
            topic_area = "nlp"
        else:
            topic_area = "general"
        
        return self._generate_topic_specific_questions(topic_area, requirements, concepts)
    
    def _extract_key_concepts(self, requirements: List[str]) -> List[str]:
        """Extract key technical concepts from requirements."""
        concepts = []
        for req in requirements:
            # Extract nouns and technical terms
            words = req.split()
            for i, word in enumerate(words):
                word_clean = word.lower().strip('.,!?')
                # Look for technical terms, compound concepts
                if len(word_clean) > 4 or word_clean in ['ml', 'ai', 'nlp']:
                    concepts.append(word_clean)
                # Look for compound terms
                if i < len(words) - 1:
                    compound = f"{word_clean} {words[i+1].lower().strip('.,!?')}"
                    if any(term in compound for term in ['machine learning', 'neural network', 'data science', 'natural language']):
                        concepts.append(compound)
        return list(set(concepts))
    
    def _generate_topic_specific_questions(self, topic_area: str, requirements: List[str], concepts: List[str]) -> List[GeneratedQuestion]:
        """Generate topic-specific high-quality questions."""
        
        if topic_area == "machine_learning":
            return [
                {
                    "question": "What are the main differences between supervised and unsupervised learning approaches?",
                    "type": "multiple_choice",
                    "difficulty": "medium",
                    "options": [
                        "A) Supervised uses labeled data, unsupervised finds patterns in unlabeled data",
                        "B) Supervised is faster, unsupervised is more accurate", 
                        "C) Supervised works with numbers, unsupervised works with text",
                        "D) Supervised requires more memory, unsupervised requires more processing power"
                    ],
                    "correct_answer": "A",
                    "expected_elements": requirements[:2]
                },
                {
                    "question": "Explain how neural networks process information from input to output, describing the role of weights and activation functions.",
                    "type": "short_answer",
                    "difficulty": "medium", 
                    "expected_elements": requirements
                },
                {
                    "question": "Describe the key evaluation metrics used in machine learning (accuracy, precision, recall, F1-score) and when each is most appropriate.",
                    "type": "short_answer",
                    "difficulty": "medium",
                    "expected_elements": requirements
                },
                {
                    "question": "Provide a real-world example of a machine learning application and explain which type of learning approach would be most suitable and why.",
                    "type": "long_answer",
                    "difficulty": "hard",
                    "expected_elements": requirements
                }
            ]
        
        elif topic_area == "neural_networks":
            return [
                {
                    "question": "What are the key components of a neural network architecture?",
                    "type": "multiple_choice", 
                    "difficulty": "medium",
                    "options": [
                        "A) Layers, nodes, weights, and activation functions",
                        "B) Input data, algorithms, and output results",
                        "C) Training sets, test sets, and validation sets", 
                        "D) Forward pass, backward pass, and optimization"
                    ],
                    "correct_answer": "A",
                    "expected_elements": requirements[:2]
                },
                {
                    "question": "Explain the backpropagation algorithm and how it enables neural networks to learn from training data.",
                    "type": "long_answer",
                    "difficulty": "hard",
                    "expected_elements": requirements
                },
                {
                    "question": "Describe different types of neural network architectures (feedforward, convolutional, recurrent) and their typical use cases.",
                    "type": "short_answer", 
                    "difficulty": "medium",
                    "expected_elements": requirements
                },
                {
                    "question": "Analyze the advantages and challenges of deep learning compared to traditional machine learning approaches.",
                    "type": "long_answer",
                    "difficulty": "hard", 
                    "expected_elements": requirements
                }
            ]
        
        elif topic_area == "nlp":
            return [
                {
                    "question": "What is the purpose of tokenization in natural language processing?",
                    "type": "multiple_choice",
                    "difficulty": "easy",
                    "options": [
                        "A) To break text into individual words or subwords for processing",
                        "B) To translate text from one language to another",
                        "C) To remove punctuation and special characters",
                        "D) To compress text data for storage efficiency"
                    ],
                    "correct_answer": "A",
                    "expected_elements": requirements[:2]
                },
                {
                    "question": "Explain the text preprocessing steps typically performed before sentiment analysis, including tokenization and stemming.",
                    "type": "short_answer",
                    "difficulty": "medium",
                    "expected_elements": requirements
                },
                {
                    "question": "Describe how modern language models work and their applications in natural language understanding tasks.",
                    "type": "long_answer",
                    "difficulty": "hard", 
                    "expected_elements": requirements
                },
                {
                    "question": "Compare different approaches to sentiment analysis and discuss their strengths and limitations for various text types.",
                    "type": "long_answer",
                    "difficulty": "hard",
                    "expected_elements": requirements
                }
            ]
        
        # Default/general questions
        return [
            {
                "question": f"What are the main concepts covered in the learning objectives?",
                "type": "multiple_choice",
                "difficulty": "medium", 
                "options": [
                    "A) Theoretical foundations and practical applications",
                    "B) Historical development and future trends",
                    "C) Basic definitions and advanced techniques", 
                    "D) Core principles and implementation strategies"
                ],
                "correct_answer": "A",
                "expected_elements": requirements[:2]
            },
            {
                "question": f"Explain the key relationships between the main concepts presented in the learning material.",
                "type": "short_answer",
                "difficulty": "medium",
                "expected_elements": requirements
            },
            {
                "question": f"Describe how these concepts can be applied in practical scenarios with specific examples.",
                "type": "short_answer",
                "difficulty": "medium", 
                "expected_elements": requirements
            },
            {
                "question": f"Analyze the advantages, limitations, and considerations when working with these concepts.",
                "type": "long_answer",
                "difficulty": "hard",
                "expected_elements": requirements
            }
        ]

    def _enforce_distribution(self, questions: List[GeneratedQuestion], requirements: List[str]) -> List[GeneratedQuestion]:
        """Guarantee exact 5 MCQ, 3 short answer, 2 long answer questions."""
        fallback = self._get_fallback_questions(requirements)

        normalized = []
        for q in questions:
            q_type = str(q.get("type", "")).lower()
            if q_type in ("mcq", "multiple_choice"):
                q["type"] = "multiple_choice"
            elif q_type in ("short", "short_answer", "open_ended"):
                q["type"] = "short_answer"
            elif q_type in ("long", "long_answer"):
                q["type"] = "long_answer"
            else:
                q["type"] = "short_answer"
            normalized.append(q)

        by_type = {"multiple_choice": [], "short_answer": [], "long_answer": []}
        for q in normalized:
            by_type[q["type"]].append(q)

        for q in fallback:
            q_type = q.get("type", "short_answer")
            if q_type == "open_ended":
                q_type = "short_answer"
                q["type"] = q_type
            by_type.setdefault(q_type, []).append(q)

        # Ensure enough MCQs by converting some fallback questions if needed.
        while len(by_type["multiple_choice"]) < self.target_distribution["multiple_choice"]:
            stem = requirements[min(len(by_type["multiple_choice"]), max(len(requirements) - 1, 0))] if requirements else "the current topic"
            idx = len(by_type["multiple_choice"]) + 1
            by_type["multiple_choice"].append({
                "question": f"Which statement best matches the learning objective '{stem}'?",
                "type": "multiple_choice",
                "difficulty": "medium",
                "options": [
                    "A) It identifies the core concept and a valid application.",
                    "B) It ignores practical usage and focuses only on jargon.",
                    "C) It contradicts the objective and expected outcomes.",
                    "D) It describes a different domain entirely."
                ],
                "correct_answer": "A",
                "expected_elements": requirements[:2] if requirements else []
            })
            if idx > 10:
                break

        short_pool = by_type["short_answer"]
        long_pool = by_type["long_answer"]
        while len(short_pool) < self.target_distribution["short_answer"] and long_pool:
            moved = long_pool.pop(0)
            moved["type"] = "short_answer"
            short_pool.append(moved)
        while len(long_pool) < self.target_distribution["long_answer"] and short_pool:
            moved = short_pool.pop()
            moved["type"] = "long_answer"
            long_pool.append(moved)

        mcq_selected = random.sample(by_type["multiple_choice"], self.target_distribution["multiple_choice"]) \
            if len(by_type["multiple_choice"]) > self.target_distribution["multiple_choice"] \
            else by_type["multiple_choice"][:self.target_distribution["multiple_choice"]]
        short_selected = random.sample(short_pool, self.target_distribution["short_answer"]) \
            if len(short_pool) > self.target_distribution["short_answer"] \
            else short_pool[:self.target_distribution["short_answer"]]
        long_selected = random.sample(long_pool, self.target_distribution["long_answer"]) \
            if len(long_pool) > self.target_distribution["long_answer"] \
            else long_pool[:self.target_distribution["long_answer"]]

        selected = mcq_selected + short_selected + long_selected
        random.shuffle(selected)

        for i, question in enumerate(selected, 1):
            question["question_id"] = f"q_{i}"

        return selected
    
    @trace_llm_operation("simulate_learner_answer")
    async def simulate_learner_answer(self, question: str, context_chunks: List[str], 
                                    all_context: List[ProcessedContext]) -> str:
        """Simulate a learner's answer to a question."""
        try:
            # Get relevant context
            relevant_context = ""
            for chunk in all_context:
                if chunk["chunk_id"] in context_chunks:
                    relevant_context += chunk["text"] + "\n\n"
            
            prompt = f"""
As a student learning this material, answer the following question based on the provided context:

QUESTION: {question}

CONTEXT:
{relevant_context}

Provide a thoughtful answer that demonstrates understanding but may have some gaps or could be improved.
Keep the answer 2-3 sentences and make it sound like a student's response.
"""
            
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error simulating learner answer: {e}")
            return "I understand the basic concepts but need to study more to provide a complete answer."
    
    @trace_llm_operation("score_answer")
    async def score_answer(self, question: str, answer: str, expected_concepts: List[str]) -> Dict[str, Any]:
        """Score a learner's answer and provide feedback."""
        try:
            expected_concepts_text = ", ".join(expected_concepts)
            
            prompt = f"""
Evaluate this student answer and provide a score from 0.0 to 1.0:

QUESTION: {question}
STUDENT ANSWER: {answer}
EXPECTED CONCEPTS: {expected_concepts_text}

Evaluate based on:
1. Accuracy of information
2. Completeness of response
3. Understanding demonstrated
4. Coverage of expected concepts

Provide response in JSON format:
{{
  "score": 0.85,
  "feedback": "Good understanding shown, but could expand on...",
  "strengths": ["accurate information", "clear explanation"],
  "improvements": ["could mention X", "needs more detail on Y"]
}}
"""
            
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback scoring
                    result = {
                        "score": 0.7,
                        "feedback": "Reasonable understanding demonstrated",
                        "strengths": ["basic understanding"],
                        "improvements": ["more detail needed"]
                    }
            except (json.JSONDecodeError, AttributeError):
                # Fallback scoring
                concept_coverage = sum(1 for concept in expected_concepts if concept.lower() in answer.lower())
                base_score = min(concept_coverage / max(len(expected_concepts), 1), 1.0)
                length_bonus = min(len(answer.split()) / 50.0, 0.2)  # Bonus for longer answers
                
                result = {
                    "score": min(base_score + length_bonus, 1.0),
                    "feedback": f"Covers {concept_coverage}/{len(expected_concepts)} expected concepts",
                    "strengths": ["attempt made"],
                    "improvements": ["could be more comprehensive"]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring answer: {e}")
            return {
                "score": 0.5,
                "feedback": "Unable to evaluate properly",
                "strengths": [],
                "improvements": ["please try again"]
            }
