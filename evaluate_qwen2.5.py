import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import random
import numpy as np
from peft import PeftModel


# Load the Qwen model and tokenizer

print("--- Starting Code")

model_name = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, "ybian-umd/Qwen2.5-7B-Instruct-gsm8k-2")

print("--- Loaded Model")

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

def get_answer(model, tokenizer, question):
    """Get model's response to a question with explanation and final answer"""
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
            num_beams=5,         # Increased for better reasoning
            do_sample=False,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Remove the prompt tokens from the output
    prompt_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][prompt_length:]
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
    
    return response

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
correct = 0
total = 0
errors = 0
differences = []

# For each question, get the answer
for i, sample in enumerate(selected_samples):
    question = sample['question']
    true_answer = extract_number(sample['answer'])
    response = get_answer(model, tokenizer, question)
    
    # Extract model's answer
    model_answer = extract_number(response.replace('###', '####'))  # Convert ### to #### for consistency
    
    # Compare answers
    is_correct = False
    if model_answer is not None and true_answer is not None:
        # Check if answers are equal within a small tolerance
        relative_error = abs(model_answer - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
        is_correct = relative_error < 0.01  # 1% tolerance
        differences.append(relative_error)
        total += 1
        if is_correct:
            correct += 1
    else:
        errors += 1
    
    # Print results
    print(f"\nQuestion {i+1}/100:")
    print(f"Question: {question}")
    print(f"True answer: {true_answer}")
    print(f"Model response:\n{response}")
    print(f"Model answer extracted: {model_answer}")
    print(f"Correct: {is_correct}")
    print("\n" + "="*50 + "\n")
    
    # Print running accuracy every 10 questions
    if (i + 1) % 10 == 0:
        print(f"Running accuracy after {i+1} questions: {correct/total:.2%}")

# Final statistics
print("\nFinal Results:")
print(f"Total questions: {len(selected_samples)}")
print(f"Successfully evaluated: {total}")
print(f"Parsing errors: {errors}")
print(f"Correct answers: {correct}")
print(f"Accuracy: {correct/total:.2%}")

if differences:
    print(f"Mean relative error: {np.mean(differences):.4f}")
    print(f"Median relative error: {np.median(differences):.4f}")