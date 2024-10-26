# init_debate.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Tuple, List, Optional
import logging
import random
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebateModel:
    def __init__(self, model_path: str, model_name: str):
        """Initialize a debate participant model."""
        self.model_name = model_name
        try:
            # Adjust the path to be relative to the project root
            full_path = "/fs/classhomes/fall2024/cmsc473/c4730005/w2s/models/weak/checkpoint-350"
            self.tokenizer = AutoTokenizer.from_pretrained(full_path)
            self.model = AutoModelForCausalLM.from_pretrained(full_path)
            logger.info(f"Successfully loaded model: {model_name} from {full_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

    def generate_answer(self, question: str, options: str) -> str:
        """Generate an answer for a given question."""
        try:
            prompt = f"Question: {question}\nOptions: {options}\nAnswer: "
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            outputs = self.model.generate(
                **inputs,
                max_length=100,
                num_beams=4,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                no_repeat_ngram_size=2
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._extract_option(answer, options)
        except Exception as e:
            logger.error(f"Error generating answer for {self.model_name}: {str(e)}")
            return "Error generating answer"

    def generate_argument(self, question: str, options: str, current_answer: str, 
                         other_answers: Dict[str, str]) -> str:
        """Generate an argument defending the model's answer."""
        try:
            argument_prompt = (
                f"Question: {question}\n"
                f"Options: {options}\n"
                f"Your answer: {current_answer}\n"
                f"Other answers: {', '.join(other_answers.values())}\n"
                f"Explain why your answer is correct: "
            )
            
            inputs = self.tokenizer(argument_prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_beams=4,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating argument for {self.model_name}: {str(e)}")
            return "Error generating argument"

    @staticmethod
    def _extract_option(answer: str, options: str) -> str:
        """Extract the multiple choice option from the model's answer."""
        valid_options = [opt.strip() for opt in options.split(")")]
        answer = answer.upper()
        for option in valid_options:
            if option and option[0] in "ABCD" and option[0] in answer:
                return option[0]
        return random.choice("ABCD")  # Fallback to random if no valid option found

class DebateArena:
    def __init__(self, num_models: int = 2):
        """Initialize the debate arena with multiple models."""
        self.models: Dict[str, DebateModel] = {}
        base_path = os.path.join(os.getcwd(), "models", "weak", "checkpoint-350")
        
        for i in range(num_models):
            model_name = f"weak_model_{i+1}"
            try:
                self.models[model_name] = DebateModel(base_path, model_name)
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {str(e)}")

    def conduct_debate(self, question: str, options: str, rounds: int = 2) -> Dict[str, str]:
        """Conduct a debate among the models."""
        if not self.models:
            logger.error("No models available for debate")
            return {}

        # Initial answers
        answers = {}
        logger.info("\n=== Initial Answers ===")
        for name, model in self.models.items():
            answers[name] = model.generate_answer(question, options)
            logger.info(f"{name}: {answers[name]}")

        # Check for consensus
        unique_answers = set(answers.values())
        if len(unique_answers) == 1:
            logger.info("\nConsensus reached! All models agree on answer: " + next(iter(unique_answers)))
            return answers

        # Debate rounds
        for round_num in range(rounds):
            logger.info(f"\n=== Debate Round {round_num + 1} ===")
            for name, model in self.models.items():
                other_answers = {k: v for k, v in answers.items() if k != name}
                argument = model.generate_argument(
                    question, 
                    options, 
                    answers[name], 
                    other_answers
                )
                logger.info(f"\n{name}'s argument:")
                logger.info(argument)

            # Allow models to change their answers based on arguments
            new_answers = {}
            for name, model in self.models.items():
                new_answer = model.generate_answer(question, options)
                if new_answer != answers[name]:
                    logger.info(f"\n{name} changed answer from {answers[name]} to {new_answer}")
                new_answers[name] = new_answer
            answers = new_answers

        return answers

def main():
    question = "What is 5 + 5?"
    options = "A) 1 B) 10 C) 100 D) 9"
    
    try:
        arena = DebateArena(num_models=2)
        final_answers = arena.conduct_debate(question, options)
        
        logger.info("\n=== Final Results ===")
        for name, answer in final_answers.items():
            logger.info(f"{name}'s final answer: {answer}")
            
    except Exception as e:
        logger.error(f"Error in debate process: {str(e)}")

if __name__ == "__main__":
    main()