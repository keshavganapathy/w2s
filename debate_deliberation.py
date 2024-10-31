import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import random
import numpy as np
from peft import PeftModel

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to extract numerical answer from text
def extract_number(text):
    try:
        # Split by #### or ### and get the last part
        parts = re.split(r'####|###', text)
        if len(parts) < 2:
            return None
        after_hash = parts[-1].strip()
        # Find the last number in the text (including negative and decimal)
        numbers = re.findall(r'-?\d*\.?\d+', after_hash)
        if numbers:
            return float(numbers[-1])
    except Exception as e:
        print(f"Error in extract_number: {str(e)}")
        return None
    return None

# Function to construct the initial prompt
def get_prompt(question, instruction):
    prompt = f"System: {instruction}\nUser: Question: {question}\nAssistant:\n"
    return prompt

# Function to construct the deliberation prompt
def get_deliberation_prompt(question, initial_instruction, previous_responses, previous_answer):
    prompt = f"System: {initial_instruction}\nUser: Question: {question}\n"
    prompt += f"Your previous answer was {previous_answer}.\n"
    prompt += "Here are two answers to the question:\n"
    for idx, response in enumerate(previous_responses):
        prompt += f"Assistant {idx+1}: {response}\n"
    prompt += "Use this information to see if you want to switch your answer from your previous answer.\n"
    prompt += """Remember to Explain your solution step by step, then provide the final numerical answer after '###'.

        Your response should be in this format:
        Step-by-step explanation...
        ###
        numerical_answer"""
    return prompt

# Function to get the model's answer
def get_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=5,
            do_sample=False,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    # Remove the prompt tokens from the output
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

# Load the dataset
print("--- Load Dataset")
dataset = data_preparation(difficulty=-1)
print("--- Select Datasets")
random.seed(10)
indices = random.sample(range(len(dataset)), 10)
selected_samples = dataset.select(indices)
print("--- Selected Datasets")

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


print("--- Starting For Loop")
# Initialize tracking variables
correct_initial = 0
correct_final = 0
total = 0
differences = []

n_rounds = 1  # Number of deliberation rounds

initial_instruction = (
    "Explain your solution step by step, then provide the final numerical answer after '###'.\n"
    "Your response should be in this format:\n"
    "Step-by-step explanation...\n"
    "###\n"
    "numerical_answer"
)

for i, sample in enumerate(selected_samples):
    question = sample['question']
    true_answer = extract_number(sample['answer'])

    # Initial answers from both sides (same model)
    prompt = get_prompt(question, initial_instruction)
    response1 = get_answer(model, tokenizer, prompt)
    response2 = get_answer(model, tokenizer, prompt)

    # Store initial responses
    initial_response1 = response1
    initial_response2 = response2

    # Extract initial answers
    initial_model_answer1 = extract_number(response1)
    initial_model_answer2 = extract_number(response2)

    model_answer1 = initial_model_answer1
    model_answer2 = initial_model_answer2

    # Deliberation rounds
    for round_num in range(n_rounds):
        # For model 1
        previous_responses = [initial_response1, initial_response2]
        prompt1 = get_deliberation_prompt(question, initial_instruction, previous_responses, model_answer1)
        response1_new = get_answer(model, tokenizer, prompt1)
        model_answer1_new = extract_number(response1_new)

        # For model 2
        previous_responses = [initial_response2, initial_response1]
        prompt2 = get_deliberation_prompt(question, initial_instruction, previous_responses, model_answer2)
        response2_new = get_answer(model, tokenizer, prompt2)
        model_answer2_new = extract_number(response2_new)

        # Update responses and model answers
        response1 = response1_new
        model_answer1 = model_answer1_new

        response2 = response2_new
        model_answer2 = model_answer2_new

    # Evaluate initial answers
    is_correct_initial1 = False
    is_correct_initial2 = False
    if initial_model_answer1 is not None and true_answer is not None:
        relative_error = abs(initial_model_answer1 - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
        is_correct_initial1 = relative_error < 0.01  # 1% tolerance
    if initial_model_answer2 is not None and true_answer is not None:
        relative_error = abs(initial_model_answer2 - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
        is_correct_initial2 = relative_error < 0.01  # 1% tolerance

    # Evaluate final answers
    is_correct_final1 = False
    is_correct_final2 = False
    if model_answer1 is not None and true_answer is not None:
        relative_error = abs(model_answer1 - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
        is_correct_final1 = relative_error < 0.01  # 1% tolerance
        differences.append(relative_error)
    if model_answer2 is not None and true_answer is not None:
        relative_error = abs(model_answer2 - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
        is_correct_final2 = relative_error < 0.01  # 1% tolerance
        differences.append(relative_error)

    total += 1
    if is_correct_initial1:
        correct_initial += 1
    if is_correct_final1:
        correct_final += 1
    if is_correct_initial2:
        correct_initial += 1
    if is_correct_final2:
        correct_final += 1

    # Print results
    print(f"\nQuestion {i+1}/100:")
    print(f"Question: {question}")
    print(f"True answer: {true_answer}")
    print(f"\nModel 1 initial response:\n{initial_response1}")
    print(f"\nModel 2 initial response:\n{initial_response2}")
    print(f"\nModel 1 response after seeing responses:\n{response1}")
    print(f"\nModel 2 response after seeing responses:\n{response2}")
    print(f"\nFinal extracted answer model 1: {model_answer1}")
    print(f"Final extracted answer model 2: {model_answer2}")
    print(f"Changed answer model 1: {initial_model_answer1 != model_answer1}")
    print(f"Changed answer model 2: {initial_model_answer2 != model_answer2}")
    print(f"Initial correct model 1: {is_correct_initial1}")
    print(f"Final correct model 1: {is_correct_final1}")
    print(f"Initial correct model 2: {is_correct_initial2}")
    print(f"Final correct model 2: {is_correct_final2}")
    print("\n" + "="*50 + "\n")

    # Print running accuracy every 10 questions
    if (i + 1) % 10 == 0:
        print(f"After {i+1} questions:")
        print(f"Total initial correct answers: {correct_initial}/{2*total} ({(correct_initial)/(2*total):.2%})")
        print(f"Total final correct answers: {correct_final}/{2*total} ({(correct_final)/(2*total):.2%})")

# Final statistics
print("\nFinal Results:")
print(f"Total questions: {total}")
print(f"Total initial correct answers: {correct_initial}/{2*total} ({(correct_initial)/(2*total):.2%})")
print(f"Total final correct answers: {correct_final}/{2*total} ({(correct_final)/(2*total):.2%})")
if differences:
    print(f"Mean relative error: {np.mean(differences):.4f}")
    print(f"Median relative error: {np.median(differences):.4f}")
