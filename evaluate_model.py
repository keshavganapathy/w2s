# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import random
import re
import numpy as np
import csv  # Added for CSV writing

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

def load_model(path):
    """Load model and tokenizer from path"""
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def get_answer(model, tokenizer, question):
    """Get model's response to a question with explanation and final answer"""
    # Construct prompt requesting explanation and final answer format
    prompt = (
        f"Question: {question}\n"
        "Explain your solution step by step, then provide the final numerical answer after the ###.\n"
        "Your response should be in this format:\n"
        "Step-by-step explanation...\n"
        "###\n"
        "numerical_answer\n\n"
        "Answer: "
    )
    
    inputs = tokenizer(prompt, 
                      return_tensors="pt", 
                      truncation=True, 
                      max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,  # Increased to allow for explanation + answer
            num_beams=5,     # Increased for better reasoning
            do_sample=False,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Ensure response has the ### separator
    if "###" not in response:
        # Try to extract the last number as the answer
        numbers = re.findall(r'-?\d*\.?\d+', response)
        if numbers:
            answer = numbers[-1]
            explanation = response.strip()
            response = f"{explanation}\n###\n{answer}"
        else:
            response = f"{response}\n###\nERROR: No number found"
    
    return response

def extract_number(text):
    """Extract numerical answer from text"""
    try:
        # Split by #### and get the last part
        parts = text.split('####')
        if len(parts) < 2:
            return None
                
        after_hash = parts[-1].strip()
        
        # Find the last number in the text (including negative and decimal)
        numbers = re.findall(r'-?\d*\.?\d+', after_hash)
        if numbers:
            # Take the last number found after ####
            return float(numbers[-1])
    except Exception as e:
        print(f"Error in extract_number: {str(e)}")
        return None
    return None

def evaluate_accuracy(predictions, ground_truth, tolerance=1e-6):
    """Calculate accuracy with a small tolerance for floating-point differences"""
    correct = 0
    total = len(predictions)
    
    for pred, true in zip(predictions, ground_truth):
        if pred is not None and true is not None and abs(pred - true) < tolerance:
            correct += 1
                
    return correct / total if total > 0 else 0

def main():
    # Model paths
    model_paths = [
        "/fs/classhomes/fall2024/cmsc473/c4730005/w2s/models/strong/checkpoint-33624",
    ]
    
    # Load dataset
    dataset = data_preparation(-1)  # Get full dataset
    
    # Select 100 random problems
    num_samples = 100
    random_indices = random.sample(range(len(dataset)), num_samples)
    test_subset = dataset.select(random_indices)
    
    # Test each model
    for i, path in enumerate(model_paths, 1):
        print(f"\nEvaluating Model {i} from {path}")
        try:
            model, tokenizer = load_model(path)
            print("Model loaded successfully")
            
            predictions = []
            ground_truth = []
            data_rows = []  # List to store data for CSV
            
            for idx, example in enumerate(test_subset):
                question = example['question']
                true_answer = extract_number(example['answer'])  # Extract numerical answer
                
                print(f"\nProblem {idx+1}/{num_samples}")
                print(f"Question: {question}")
                print(f"True answer: {true_answer}")
                
                try:
                    response = get_answer(model, tokenizer, question)
                    pred_answer = extract_number(response)
                    
                    print(f"Model response: {response}")
                    print(f"Extracted answer: {pred_answer}")
                    
                    predictions.append(pred_answer)
                    ground_truth.append(true_answer)
                    
                    # Collect data for CSV
                    data_rows.append({
                        'Question': question,
                        'Ground Truth Answer': true_answer,
                        'Model Response': response,
                        'Extracted Answer': pred_answer
                    })
                    
                except Exception as e:
                    print(f"Error processing question: {str(e)}")
                    predictions.append(None)
                    ground_truth.append(true_answer)
                    
                    # Collect data for CSV with None values
                    data_rows.append({
                        'Question': question,
                        'Ground Truth Answer': true_answer,
                        'Model Response': 'Error',
                        'Extracted Answer': None
                    })
            
            # Calculate and display metrics
            accuracy = evaluate_accuracy(predictions, ground_truth)
            response_rate = sum(1 for p in predictions if p is not None) / len(predictions)
            
            print("\nEvaluation Results:")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Response Rate: {response_rate:.2%}")
            
            # Additional statistics
            valid_predictions = [p for p in predictions if p is not None]
            if valid_predictions:
                mean_pred = np.mean(valid_predictions)
                std_pred = np.std(valid_predictions)
                print(f"Mean prediction: {mean_pred:.2f}")
                print(f"Std deviation: {std_pred:.2f}")
            
            # Write data to CSV
            csv_file_name = f"model_{i}_results.csv"
            with open(csv_file_name, mode='w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Question', 'Ground Truth Answer', 'Model Response', 'Extracted Answer']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in data_rows:
                    writer.writerow(row)
            
            print(f"Results saved to {csv_file_name}")
            
        except Exception as e:
            print(f"Error with model {i}: {str(e)}")

if __name__ == "__main__":
    main()
