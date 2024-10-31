import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import random
import numpy as np
from peft import PeftModel
import matplotlib.pyplot as plt

print("--- Starting Code")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, "ybian-umd/Qwen2.5-3B-Instruct-gsm8k-6")

print("--- Loaded Model")

def extract_number(text):
    """Extract numerical answer from text"""
    try:
        # Remove any whitespace or irrelevant characters
        text = text.strip()
        
        # Find all occurrences of '###' and split the text
        parts = re.split(r'###', text)
        
        # If less than 2 parts, the format is incorrect
        if len(parts) < 2:
            return None

        # The final answer should be in the last non-empty part
        for part in reversed(parts):
            part = part.strip()
            if part:
                # Find all numbers in the part
                numbers = re.findall(r'-?\d*\.?\d+', part)
                if numbers:
                    # Return the first number found
                    return float(numbers[0])
                else:
                    continue
        return None
    except Exception as e:
        print(f"Error in extract_number: {str(e)}")
        return None

# Enhanced prompt construction with more explicit formatting
def get_prompt(question):
    prompt = (
        f"System: You are a math problem solver. Follow these instructions exactly:\n"
        f"1. Solve the problem step by step.\n"
        f"2. After your solution, add a line containing exactly three hash symbols (###).\n"
        f"3. On the next line, write only the final numerical answer without any text.\n"
        f"4. Do not add any text after the numerical answer.\n\n"
        f"Example format:\n"
        f"Step 1: First calculation...\n"
        f"Step 2: Second calculation...\n"
        f"Therefore, the answer is calculated.\n"
        f"###\n"
        f"42\n\n"
        f"Now solve this question:\n\n"
        f"{question}\n"
        f"Assistant:"
    )
    return prompt

# Enhanced deliberation prompt with format reinforcement
def get_deliberation_prompt(question, previous_responses, previous_answer):
    # Remove '###' from previous responses to avoid confusing the model
    cleaned_responses = []
    for response in previous_responses:
        cleaned_response = response.replace('###', '').strip()
        cleaned_responses.append(cleaned_response)

    prompt = (
        f"System: You are a math problem solver reviewing previous answers. Follow these instructions exactly:\n"
        f"1. Review the previous responses and your previous answer ({previous_answer}).\n"
        f"2. Analyze the approaches and determine the correct solution.\n"
        f"3. Provide your step-by-step solution.\n"
        f"4. End with '###' on a new line.\n"
        f"5. On the next line, write only the final numerical answer without any text.\n"
        f"6. Do not add any text after the numerical answer.\n\n"
        f"Question:\n\n"
        f"{question}\n\n"
        f"Previous responses:\n"
    )
    for idx, response in enumerate(cleaned_responses):
        prompt += f"Response {idx+1}:\n{response}\n\n"

    prompt += (
        f"Provide your revised solution following the exact format specified."
    )
    return prompt

# Modified generation parameters to encourage format compliance
def get_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,      # Increased to allow for explanation + answer
            do_sample=True,           # Enable sampling
            temperature=0.7,          # Adjust temperature for randomness
            top_p=0.9,                # Use nucleus sampling
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    prompt_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()

# Data preparation function
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

# Function to get bin label based on rating_quantile
def get_bin_label(rating_quantile, difficulty_bins, bin_labels):
    for i in range(len(difficulty_bins)-1):
        if difficulty_bins[i] <= rating_quantile < difficulty_bins[i+1]:
            return bin_labels[i]
    return bin_labels[-1]  # In case it's exactly 1.0

# Initialize bin counts
def initialize_bin_counts(bin_labels):
    bin_counts = {}
    for bin_label in bin_labels:
        bin_counts[bin_label] = {
            'total': 0,
            'correct_initial1': 0,
            'correct_final1': 0,
            'correct_initial2': 0,
            'correct_final2': 0
        }
    return bin_counts

# Load the dataset
print("--- Load Dataset")
dataset = data_preparation(difficulty=-1)

# === Modification Starts Here ===

# # Define difficulty bins
difficulty_bins = np.linspace(0, 1, num=6)  # 5 bins
bin_labels = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']

# Initialize bin counts
bin_counts = initialize_bin_counts(bin_labels)

# Number of questions per bin (modifiable variable)
QUESTIONS_PER_BIN = 10

# Function to process a single bin
def process_bin(bin_label, bin_dataset, n_rounds, bin_counts):
    print(f"Processing Bin: {bin_label}")
    
    # Check if bin_dataset has enough samples
    num_available = len(bin_dataset)
    num_samples = min(QUESTIONS_PER_BIN, num_available)
    if num_samples < QUESTIONS_PER_BIN:
        print(f"Warning: Only {num_samples} samples available for bin '{bin_label}'")
    
    if num_samples == 0:
        print(f"No data available for bin '{bin_label}'. Skipping...")
        return bin_counts
    
    # Randomly sample QUESTIONS_PER_BIN questions from bin_dataset
    random.seed(10)
    sampled_indices = random.sample(range(len(bin_dataset)), num_samples)
    sampled_dataset = bin_dataset.select(sampled_indices)
    
    print(f"Number of questions in '{bin_label}' bin: {num_samples}")
    
    # Initialize question counter for the bin
    question_counter = 1
    
    # For each question in sampled_dataset
    for idx, sample in enumerate(sampled_dataset):
        question = sample['question']
        true_answer = extract_number(sample['answer'])
        
        # Print question identifier
        print(f"{bin_label} - {question_counter}")
        question_counter += 1
        
        # Initial answers from both sides (same model)
        prompt = get_prompt(question)
        response1 = get_answer(model, tokenizer, prompt)
        response2 = get_answer(model, tokenizer, prompt)

        # Extract initial answers
        initial_model_answer1 = extract_number(response1)
        initial_model_answer2 = extract_number(response2)

        model_answer1 = initial_model_answer1
        model_answer2 = initial_model_answer2

        # Deliberation rounds
        for round_num in range(n_rounds):
            # For model 1
            previous_responses = [response1, response2]
            prompt1 = get_deliberation_prompt(question, previous_responses, model_answer1)
            response1_new = get_answer(model, tokenizer, prompt1)
            model_answer1_new = extract_number(response1_new)

            # For model 2
            previous_responses = [response2, response1]
            prompt2 = get_deliberation_prompt(question, previous_responses, model_answer2)
            response2_new = get_answer(model, tokenizer, prompt2)
            model_answer2_new = extract_number(response2_new)

            # Update model answers
            model_answer1 = model_answer1_new
            model_answer2 = model_answer2_new

        # Evaluate initial answers
        is_correct_initial1 = False
        is_correct_initial2 = False
        if initial_model_answer1 is not None and true_answer is not None:
            relative_error1 = abs(initial_model_answer1 - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
            is_correct_initial1 = relative_error1 < 0.01  # 1% tolerance
        if initial_model_answer2 is not None and true_answer is not None:
            relative_error2 = abs(initial_model_answer2 - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
            is_correct_initial2 = relative_error2 < 0.01  # 1% tolerance

        # Evaluate final answers
        is_correct_final1 = False
        is_correct_final2 = False
        if model_answer1 is not None and true_answer is not None:
            relative_error_final1 = abs(model_answer1 - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
            is_correct_final1 = relative_error_final1 < 0.01  # 1% tolerance
        if model_answer2 is not None and true_answer is not None:
            relative_error_final2 = abs(model_answer2 - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
            is_correct_final2 = relative_error_final2 < 0.01  # 1% tolerance

        # Update bin counts
        bin_counts[bin_label]['total'] += 1
        if is_correct_initial1:
            bin_counts[bin_label]['correct_initial1'] += 1
        if is_correct_final1:
            bin_counts[bin_label]['correct_final1'] += 1
        if is_correct_initial2:
            bin_counts[bin_label]['correct_initial2'] += 1
        if is_correct_final2:
            bin_counts[bin_label]['correct_final2'] += 1

    # After processing the bin, print the accuracies
    total_bin = bin_counts[bin_label]['total']
    if total_bin > 0:
        acc_initial1 = bin_counts[bin_label]['correct_initial1'] / total_bin
        acc_final1 = bin_counts[bin_label]['correct_final1'] / total_bin
        acc_initial2 = bin_counts[bin_label]['correct_initial2'] / total_bin
        acc_final2 = bin_counts[bin_label]['correct_final2'] / total_bin
    else:
        acc_initial1 = acc_final1 = acc_initial2 = acc_final2 = 0.0

    print(f"\n--- Accuracies for '{bin_label}' Bin ---")
    print(f"Model 1 Initial Accuracy: {bin_counts[bin_label]['correct_initial1']}/{total_bin} ({acc_initial1:.2%})")
    print(f"Model 1 Final Accuracy: {bin_counts[bin_label]['correct_final1']}/{total_bin} ({acc_final1:.2%})")
    print(f"Model 2 Initial Accuracy: {bin_counts[bin_label]['correct_initial2']}/{total_bin} ({acc_initial2:.2%})")
    print(f"Model 2 Final Accuracy: {bin_counts[bin_label]['correct_final2']}/{total_bin} ({acc_final2:.2%})")
    print("\n" + "="*50 + "\n")

    return bin_counts

# Initialize tracking variables
correct_initial1 = 0
correct_initial2 = 0
correct_final1 = 0
correct_final2 = 0
total = 0
differences = []

n_rounds = 1  # Number of deliberation rounds

# Process each bin
for bin_label in bin_labels:
    # Determine the index for bin_label
    bin_index = bin_labels.index(bin_label)
    bin_start = difficulty_bins[bin_index]
    bin_end = difficulty_bins[bin_index + 1]
    
    # Filter dataset for current bin
    bin_dataset = dataset.filter(lambda example: bin_start <= example['rating_quantile'] < bin_end)
    
    # Process the bin
    bin_counts = process_bin(bin_label, bin_dataset, n_rounds, bin_counts)

# Final statistics
print("\nFinal Results:")
for bin_label in bin_labels:
    total_bin = bin_counts[bin_label]['total']
    if total_bin > 0:
        acc_initial1 = bin_counts[bin_label]['correct_initial1'] / total_bin
        acc_final1 = bin_counts[bin_label]['correct_final1'] / total_bin
        acc_initial2 = bin_counts[bin_label]['correct_initial2'] / total_bin
        acc_final2 = bin_counts[bin_label]['correct_final2'] / total_bin
    else:
        acc_initial1 = acc_final1 = acc_initial2 = acc_final2 = 0.0
    
    print(f"Bin: {bin_label}")
    print(f"  Total Questions: {total_bin}")
    print(f"  Model 1 Initial Correct: {bin_counts[bin_label]['correct_initial1']}/{total_bin} ({acc_initial1:.2%})")
    print(f"  Model 1 Final Correct: {bin_counts[bin_label]['correct_final1']}/{total_bin} ({acc_final1:.2%})")
    print(f"  Model 2 Initial Correct: {bin_counts[bin_label]['correct_initial2']}/{total_bin} ({acc_initial2:.2%})")
    print(f"  Model 2 Final Correct: {bin_counts[bin_label]['correct_final2']}/{total_bin} ({acc_final2:.2%})")
    print("\n" + "="*50 + "\n")

if differences:
    mean_rel_error = np.mean(differences)
    median_rel_error = np.median(differences)
    print(f"Mean relative error: {mean_rel_error:.4f}")
    print(f"Median relative error: {median_rel_error:.4f}")

# Plotting accuracies per bin
# Prepare data for plotting
bins = bin_labels
x = np.arange(len(bins))  # the label locations
width = 0.2  # the width of the bars

model1_initial_accuracies = []
model1_final_accuracies = []
model2_initial_accuracies = []
model2_final_accuracies = []

for bin_label in bins:
    total_bin = bin_counts[bin_label]['total']
    if total_bin > 0:
        acc_initial1 = bin_counts[bin_label]['correct_initial1'] / total_bin
        acc_final1 = bin_counts[bin_label]['correct_final1'] / total_bin
        acc_initial2 = bin_counts[bin_label]['correct_initial2'] / total_bin
        acc_final2 = bin_counts[bin_label]['correct_final2'] / total_bin
    else:
        acc_initial1 = acc_final1 = acc
