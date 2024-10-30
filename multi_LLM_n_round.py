import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import random
from load import data_preparation
from evaluate import extract_number, accuracy, best_of_n
import argparse

random.seed(10) 





def get_answer_round1(model, tokenizer, question, n=1):
    """Get model's responses to a question with explanation and final answer"""
    
    text = (
        "Explain your solution step by step to the following question, then provide the final numerical answer after the ###.\n"
        f"Question: {question}"
    )

    inputs = tokenizer([text], return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=n,
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


def multi_round_debate(models, tokenizers, question, rounds=1):
    for round in range(rounds):
        if round ==0:
            all_responses = []
            all_ans = []
            for model, tokenizer in zip(models, tokenizers):
                responses = get_answer_round1(model, tokenizer, question, n=1)
                all_responses.append(responses)
                # print(responses)
                for response in responses:
                    ans = extract_number(response)
                    print(f"Question: {question},  Answer: {ans}")
                    all_ans.append(ans)
        else:
            all_responses_new = []
            all_ans = []
            for model, tokenizer in zip(models, tokenizers):
                responses = get_answer_round1(model, tokenizer, question, n=1)
                all_responses_new.append(responses)
                # print(responses)
                for response in responses:
                    ans = extract_number(response)
                    print(f"Question: {question},  Answer: {ans}")
                    all_ans.append(ans)
            all_responses.append(all_responses_new)

    return all_ans


def main(args):
    with open(args.output_path, "a") as file:
         

    
        model_names = ["Qwen/Qwen2.5-Math-1.5B-Instruct"]
        file.write(f"Model Name : {model_names}, Number of LLm : {args.LLM_num}, Number of rounds : {args.rounds}\n") 
        print("loading models")
        models = []
        tokenizers = []
        for model_name in model_names:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                # attn_implementation="flash_attention_2"
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            models.append(model)
            tokenizers.append(tokenizer)
        
        print("loaded models")

    
        for diff in range(7):
            # Load the dataset
            dataset = data_preparation(difficulty=diff)

            indices = random.sample(range(len(dataset)), 10)
            selected_samples = dataset.select(indices)


            true_answers = []
            pred_answers = []
            for i, sample in enumerate(selected_samples):
                question = sample['question']
                true_answers.append(extract_number(sample['answer']))
                responses = multi_round_debate(models, tokenizers, question, rounds=args.rounds)

                answer = best_of_n(responses)
                pred_answers.append(answer)

            # Calculate the accuracy
            print(f"True answers: {true_answers}")
            print(f"Predicted answers: {pred_answers}")
            acc = accuracy(true_answers, pred_answers)
            print(f"Accuracy for difficulty {diff}: {acc}")
            file.write(f"Accuracy for difficulty {diff}: {acc}\n")            


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default="output/full_outputs.jsonl", type=str, help='Path to the output jsonl file where the response will be written.')
    parser.add_argument("--judge", default=False, action="store_true", help="Whether judge is involved in the debate.")
    parser.add_argument("--LLM_num", default=3, type=float, help="Total number of LLMs in the debate.")
    parser.add_argument("--rounds", default=1, type=int, help="Total number of rounds in the debate.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

        
