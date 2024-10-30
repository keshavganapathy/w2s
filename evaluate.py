import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import random
import numpy as np
from peft import PeftModel



def extract_number(text):
    """Extract numerical answer from text"""
    try:
        # Split by #### and get the last part
        parts = text.split('###')
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



def accuracy(true_answers, pred_answers):
    """Calculate the accuracy of predicted answers"""
    
    pred_correct = 0
    for pred_answer, true_answer in zip(pred_answers, true_answers):
        if pred_answer and true_answer:
            is_pred_correct = False
            relative_error = abs(pred_answer - true_answer) / (abs(true_answer) if true_answer != 0 else 1)
            is_pred_correct = relative_error < 0.01  # 1% tolerance
            if is_pred_correct:
                pred_correct +=1
        else:
            # No valid answers among model answers
            is_pred_correct = False

    return pred_correct / len(true_answers)


def best_of_n(model_answers):
    answer_counts = {}
    for ans in model_answers:
        if ans is not None:
            answer_counts[ans] = answer_counts.get(ans, 0) +1
        

    if answer_counts:
        # Get the majority answer
        best_of_n = max(answer_counts.items(), key=lambda x: x[1])[0]
    else:
        best_of_n = None

    return best_of_n

