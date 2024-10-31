import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from peft import PeftModel

print("--- Starting Code")

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

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
            max_new_tokens=256,
            num_beams=n,         # Adjusted to current n
            do_sample=False,
            temperature=0.7,
            num_return_sequences=n,
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

def evaluate_model_per_difficulty(n):
    # Set random seed for reproducibility
    random.seed(10)
    
    # Define difficulty bins
    difficulty_bins = np.linspace(0, 1, num=6)  # 5 bins
    bin_labels = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
    
    # Initialize tracking variables
    bin_results = []
    
    for i in range(len(difficulty_bins)-1):
        bin_start = difficulty_bins[i]
        bin_end = difficulty_bins[i+1]
        bin_label = bin_labels[i]
        
        print(f"Evaluating bin: {bin_label}")
        
        # Filter dataset for current bin
        bin_dataset = dataset.filter(lambda example: bin_start <= example['rating_quantile'] < bin_end)
        
        # Check if bin_dataset has enough samples
        num_samples = min(30, len(bin_dataset))
        if num_samples == 0:
            print(f"No data in bin {bin_label}")
            continue
        
        # Randomly sample 30 questions from bin_dataset
        sampled_indices = random.sample(range(len(bin_dataset)), num_samples)
        sampled_dataset = bin_dataset.select(sampled_indices)
        
        print(f"Number of questions in {bin_label} bin: {num_samples}")
        
        total = 0
        correct_counts = [0]*n
        errors = 0
        best_of_n_correct = 0
        
        # For each question in sampled_dataset
        for idx, sample in enumerate(sampled_dataset):
            question = sample['question']
            true_answer = extract_number(sample['answer'])
            responses = get_answer(model, tokenizer, question, n=n)
            
            is_corrects = []
            for idx_model, response in enumerate(responses):
                # Extract model's answer
                model_answer = extract_number(response)
                
                # Compare answers
                is_correct = False
                if model_answer is not None and true_answer is not None:
                    # Check if answers are equal within a small tolerance
                    relative_error = abs(model_answer - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
                    is_correct = relative_error < 0.01  # 1% tolerance
                    if is_correct:
                        correct_counts[idx_model] += 1
                else:
                    errors +=1
                
                is_corrects.append(is_correct)
            
            total +=1
            
            # "Best of n" accuracy
            if any(is_corrects):
                best_of_n_correct +=1
            
            # Print incremental feedback
            print(f"{bin_label} - question {total}")
        
        # Compute final accuracies
        accuracies = [correct_counts[idx]/total for idx in range(n)]
        best_of_n_accuracy = best_of_n_correct / total
        
        # Store results
        bin_results.append({
            'bin_label': bin_label,
            'bin_start': bin_start,
            'bin_end': bin_end,
            'total': total,
            'errors': errors,
            'accuracies': accuracies,
            'best_of_n_accuracy': best_of_n_accuracy
        })
    
    return bin_results

# Evaluate for n = 3
n = 5
print(f"--- Starting evaluation for n={n}")
bin_results = evaluate_model_per_difficulty(n)

# Plot accuracy vs difficulty level
import matplotlib.pyplot as plt
import numpy as np

# Extract data for plotting
bin_labels = [result['bin_label'] for result in bin_results]
x = np.arange(len(bin_labels))  # the label locations

# Accuracies per model
width = 0.15  # the width of the bars
fig, ax = plt.subplots(figsize=(12, 6))

for idx in range(n):
    model_accuracies = [result['accuracies'][idx] for result in bin_results]
    ax.bar(x + idx*width, model_accuracies, width, label=f'Model {idx+1}')

# Best of n accuracy
best_of_n_accuracies = [result['best_of_n_accuracy'] for result in bin_results]
ax.bar(x + n*width, best_of_n_accuracies, width, label='Best of n', color='gray')

# Add labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_xlabel('Difficulty Level')
ax.set_title('Accuracy vs Difficulty Level n=5')
ax.set_xticks(x + width * n / 2)
ax.set_xticklabels(bin_labels)
ax.legend()

ax.set_ylim([0, 1])  # Set y-axis limits from 0 to 1
ax.grid(True, axis='y')

fig.tight_layout()

# Save the plot
plt.savefig(f'accuracy_vs_difficulty_n{n}.png')
plt.close()
