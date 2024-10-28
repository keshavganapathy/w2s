import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import random
import numpy as np
import matplotlib.pyplot as plt

print("--- Starting Code")

# Set the number of models (n) to use
n = 5  # You can change this value to generalize for any n

# Load the Qwen model and tokenizer
model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("--- Loaded Model")

def extract_number(text):
    """Extract numerical answer from text"""
    try:
        # Split by #### and get the last part
        parts = text.split('####')
        if len(parts) < 2:
            # Try splitting by ###
            parts = text.split('###')
            if len(parts) < 2:
                return None
                    
        after_hash = parts[-1].strip()
        
        # Find the last number in the text (including negative and decimal)
        numbers = re.findall(r'-?\d*\.?\d+', after_hash)
        if numbers:
            # Take the last number found after #### or ###
            return float(numbers[-1])
    except Exception as e:
        print(f"Error in extract_number: {str(e)}")
        return None
    return None

def get_answer(model, tokenizer, question, n=1):
    """Get model's responses to a question with explanation and final answer"""
    # Construct messages
    messages = [
        {
            "role": "system",
            "content": (
                "Explain your solution step by step, then provide the final numerical answer after the ###.\n"
                "Your response should be in this format:\n"
                "Step-by-step explanation...\n"
                "###\n"
                "numerical_answer"
            )
        },
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    # Convert messages to text using apply_chat_template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Increased to allow for explanation + answer
            num_beams=1,         # Set to 1 when using sampling
            do_sample=True,      # Enable sampling
            temperature=0.7,
            top_k=50,
            num_return_sequences=n,  # Generate n sequences
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Remove the prompt tokens from the output
    prompt_length = inputs['input_ids'].shape[1]
    
    responses = []
    for output in outputs:
        generated_ids = output[prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
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
        responses.append(response)
    
    return responses

def data_preparation(difficulty=-1):
    assert difficulty >= -1 and difficulty <=6
    dataset = load_dataset(
        "furonghuang-lab/Easy2Hard-Bench",
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

print("--- Load Dataset")
# Load the dataset
dataset = data_preparation(difficulty=-1)

print("--- Select Datasets")
# Randomly select 100 questions
random.seed(42)  # For reproducibility
indices = random.sample(range(len(dataset)), 100)
selected_samples = dataset.select(indices)

print("--- Selected Datasets")

print("--- Starting For Loop")
# Initialize tracking variables
correct_counts = [0]*n  # Correct counts for each model
total = 0
errors = 0
differences = [[] for _ in range(n)]  # Differences for each model
majority_correct = 0
majority_differences = []

# For each question, get the answer
for i, sample in enumerate(selected_samples):
    question = sample['question']
    true_answer = extract_number(sample['answer'])
    responses = get_answer(model, tokenizer, question, n=n)
    
    model_answers = []
    is_corrects = []
    for idx, response in enumerate(responses):
        # Extract model's answer
        model_answer = extract_number(response)
        model_answers.append(model_answer)
        
        # Compare answers
        is_correct = False
        if model_answer is not None and true_answer is not None:
            # Check if answers are equal within a small tolerance
            relative_error = abs(model_answer - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
            is_correct = relative_error < 0.01  # 1% tolerance
            differences[idx].append(relative_error)
            if is_correct:
                correct_counts[idx] += 1
        else:
            errors +=1
        
        is_corrects.append(is_correct)
    
    total +=1
    
    # Find the majority answer among model_answers
    # Count the frequency of each answer
    answer_counts = {}
    for ans in model_answers:
        if ans is not None:
            answer_counts[ans] = answer_counts.get(ans, 0) +1
    
    if answer_counts:
        # Get the majority answer
        majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        # Compare majority answer to true answer
        is_majority_correct = False
        relative_error = abs(majority_answer - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
        is_majority_correct = relative_error < 0.01  # 1% tolerance
        majority_differences.append(relative_error)
        if is_majority_correct:
            majority_correct +=1
    else:
        # No valid answers among model answers
        is_majority_correct = False
        errors +=1
        majority_answer = None
    
    # Print results
    print(f"\nQuestion {i+1}/{len(selected_samples)}:")
    print(f"Question: {question}")
    print(f"True answer: {true_answer}")
    for idx, response in enumerate(responses):
        print(f"Model {idx+1} response:\n{response}")
        print(f"Model {idx+1} answer extracted: {model_answers[idx]}")
        print(f"Model {idx+1} Correct: {is_corrects[idx]}")
        print("-"*20)
    print(f"Majority answer: {majority_answer}")
    print(f"Majority Correct: {is_majority_correct}")
    print("\n" + "="*50 + "\n")
    
    # Print running accuracies every 10 questions
    if (i + 1) % 10 == 0:
        for idx in range(n):
            print(f"Running accuracy of Model {idx+1} after {i+1} questions: {correct_counts[idx]/total:.2%}")
        print(f"Running accuracy of Majority Vote after {i+1} questions: {majority_correct/total:.2%}")
        print("="*50)

# Final statistics
print("\nFinal Results:")
print(f"Total questions: {total}")
print(f"Parsing errors: {errors}")
for idx in range(n):
    print(f"Correct answers by Model {idx+1}: {correct_counts[idx]}")
print(f"Correct answers by Majority Vote: {majority_correct}")

for idx in range(n):
    print(f"Accuracy of Model {idx+1}: {correct_counts[idx]/total:.2%}")
print(f"Accuracy of Majority Vote: {majority_correct/total:.2%}")

# Plot accuracies
accuracies = [correct_counts[idx]/total for idx in range(n)]
labels = [f"Model {idx+1}" for idx in range(n)]
accuracies.append(majority_correct/total)
labels.append("Majority Vote")

plt.figure(figsize=(10,6))
plt.bar(labels, accuracies, color=['blue']*n + ['green'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Individual Models and Majority Vote')
plt.ylim(0,1)
plt.show()