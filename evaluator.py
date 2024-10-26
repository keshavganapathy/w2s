from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import re
from typing import List, Optional
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_preparation(difficulty:int=-1):
    assert difficulty>=-1 and difficulty<=6
    dataset = load_dataset(
            f"furonghuang-lab/Easy2Hard-Bench",
            "E2H-GSM8K",
            split="eval",
    ).select_columns(
        ["question", "answer", "rating_quantile"]
    ).sort(
        "rating_quantile"
    )
    if difficulty != -1:
        return dataset.select(range(2*difficulty*100, (2*difficulty+1)*100))
    else:
        return dataset

def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract the answer after #### from GSM8K format."""
    try:
        answer_part = text.split('####')[-1].strip()
        return float(answer_part)
    except (IndexError, ValueError):
        return None

def construct_init_query(question: str) -> str:
    """Construct a query that asks for a numerical answer."""
    return f"""Solve this math problem step by step and give the final numerical answer after '####':
{question}

Show your work with clear steps, then end with just the final numerical answer after '####'."""

class ModelEvaluator:
    def __init__(self, model_paths: List[str]):
        """Initialize models from checkpoint paths."""
        self.models = []
        self.tokenizers = []
        
        for path in model_paths:
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                logger.info(f"Successfully loaded model from {path}")
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {str(e)}")
                raise

    def generate_answer(self, model_idx: int, question: str) -> str:
        """Generate an answer using specified model."""
        try:
            model = self.models[model_idx]
            tokenizer = self.tokenizers[model_idx]
            
            prompt = construct_init_query(question)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer for model {model_idx}: {str(e)}")
            return "Error generating answer"

    def evaluate_models(self, num_samples: int = 100) -> dict:
        """Evaluate all models on GSM8K dataset."""
        dataset = data_preparation()  # Using the provided function
        results = {i: {"correct": 0, "total": 0} for i in range(len(self.models))}
        majority_results = {"correct": 0, "total": 0}
        
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
                
            question = example['question']
            correct_answer = extract_gsm8k_answer(example['answer'])
            
            # Get answers from all models
            answers = []
            for model_idx in range(len(self.models)):
                response = self.generate_answer(model_idx, question)
                extracted = extract_gsm8k_answer(response)
                
                if extracted is not None:
                    answers.append(extracted)
                    if abs(extracted - correct_answer) < 1e-6:
                        results[model_idx]["correct"] += 1
                results[model_idx]["total"] += 1
            
            # Calculate majority vote if we have answers
            if answers:
                majority_answer = max(set(answers), key=answers.count)
                if abs(majority_answer - correct_answer) < 1e-6:
                    majority_results["correct"] += 1
                majority_results["total"] += 1
                logger.info(f"Question {i+1}: Majority answer = {majority_answer}, Correct = {correct_answer}")
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1} examples")
        
        results["majority"] = majority_results
        return results

def main():
    # List your model checkpoint paths
    model_paths = [
        "/fs/classhomes/fall2024/cmsc473/c4730005/w2s/debate_models/pythia_debate_160m/models/weak/checkpoint-1869",
        "/fs/classhomes/fall2024/cmsc473/c4730005/w2s/debate_models/pythia_debate_160m/models/weak/checkpoint-3738"
    ]
    
    evaluator = ModelEvaluator(model_paths)
    results = evaluator.evaluate_models(num_samples=100)
    
    # Print results
    logger.info("\n=== Final Results ===")
    for model_idx, result in results.items():
        if model_idx == "majority":
            accuracy = result["correct"] / result["total"] if result["total"] > 0 else 0
            logger.info(f"Majority Vote accuracy: {accuracy:.4f} ({result['correct']}/{result['total']})")
        else:
            accuracy = result["correct"] / result["total"] if result["total"] > 0 else 0
            logger.info(f"Model {model_idx} accuracy: {accuracy:.4f} ({result['correct']}/{result['total']})")

if __name__ == "__main__":
    main()