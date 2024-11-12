import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import random
from load import data_preparation
from evaluate import extract_number, accuracy, best_of_n, extract_number_pred
import argparse

random.seed(10) 
torch.manual_seed(10)
torch.cuda.manual_seed(10)
torch.cuda.manual_seed_all(10)  # if you are using multi-GPU.

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def get_answer_round1(model, tokenizer, question, n=1):
    """Get model's responses to a question with explanation and final answer"""
    
    # text = (
    #     "Explain your solution step by step to the following question, then provide the final numerical answer after the ###.\n"
    #     f"Question: {question}"
    # )
    text = (
        "Provide your solution to the following question followed by final numerical answer presented inside \\boxed{{ANSWER}}"
        f"Question: {question}"
    )

    inputs = tokenizer([text], return_tensors="pt").to('cuda')
    # generator = torch.Generator(device='cuda')
    # generator.manual_seed(10)


    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=n,
            do_sample=True,
            temperature=1,
            # num_return_sequences=n,
            repetition_penalty=1,
            # pad_token_id=tokenizer.eos_token_id
            # generator=generator,
        )
    
    # Remove the prompt tokens from the output
    prompt_length = inputs['input_ids'].shape[1]
    
    responses = []
    for output in outputs:
        generated_ids = output[prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print("Answer: ", response)
        
        responses.append(response)
    
    return responses


def get_answer_roundn(model, tokenizer, question, curr_agent_response, other_agent_responses, n=1):
    """Get model's responses to a question with explanation and final answer"""
    
    text = (
        f"The question is: {question}\n."
        f"While your response is: {curr_agent_response}, other agent's response is slightly different. Do you want to change your answer? If yes, then provide then provide new solution followed by the final numerical answer presented inside \\boxed{{ANSWER}}\n"
    )

    inputs = tokenizer([text], return_tensors="pt").to('cuda')
    # generator = torch.Generator(device='cuda')
    # generator.manual_seed(10)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=n,
            do_sample=True,
            temperature=1,
            # num_return_sequences=n,
            repetition_penalty=1,
            # pad_token_id=tokenizer.eos_token_id
            # generator=generator,
        )
    
    # Remove the prompt tokens from the output
    prompt_length = inputs['input_ids'].shape[1]
    
    responses = []
    for output in outputs:
        generated_ids = output[prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print("Answer: ", response)
        
        responses.append(response)
    
    return responses


def multi_round_debate(models, tokenizers, question, rounds):
    print("Rounds: ", rounds)
    for round in range(rounds):
        if round == 0:
            print("Round: ", round)
            all_responses = []
            all_ans = []
            for model, tokenizer in zip(models, tokenizers):
                responses = get_answer_round1(model, tokenizer, question, n=1)
                all_responses.append(responses)
                # print(responses)
                for response in responses:
                    ans = extract_number_pred(response)
                
                all_ans.append(ans)
            print(f"Question: {question},  Answer: {all_ans}")
            print("\n*********\n")
        else:
            print("Round: ", round)
            all_responses_new = []
            all_ans_new = []
            numerical_values = [x for x in all_ans if x is not None]
            if len(set(all_ans)) == 1:
                print("All answers are same. No need to continue the debate.")
                print("\n*********\n")
                return all_ans
            elif len(numerical_values) >= 2 and any(numerical_values.count(x) >= 2 for x in numerical_values):
                print("Found at least two equal numerical values.")
                print("\n*********\n")
                return all_ans
            else:
                for i, (model, tokenizer) in enumerate(zip(models, tokenizers)):
                    other_responses = all_responses.copy()
                    other_responses.pop(i)
                    print("lenght of all responses: ", len(all_responses))
                    responses = get_answer_roundn(model, tokenizer, question, all_responses[i], other_responses, n=1)
                    all_responses_new.append(responses)
                    # print(responses)
                    for response in responses:
                        ans = extract_number_pred(response)
                        # if ans == None:
                            # print(f"Answer is {response}")
                    
                    all_ans_new.append(ans)
                print(f"Round: {round+1}, Question: {question},  Answer: {all_ans_new}, Old Answer: {all_ans}")
                print("\n*********\n")
            all_responses = all_responses_new
            all_ans = all_ans_new

    return all_ans


def main(args):
    with open(args.output_path, "a") as file:
         
        # model_names = ["Qwen/Qwen2.5-Math-1.5B", "monsterapi/gemma-2b-lora-maths-orca-200k", "TinyLlama/TinyLlama_v1.1_math_code", "Qwen/Qwen2.5-Math-1.5B-Instruct", "google/gemma-2-2b"]
        model_names = ["Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct"]

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
                cache_dir="models"
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="models")
            models.append(model)
            tokenizers.append(tokenizer)
        
        print("loaded models")

    
        for diff in range(5):
            # Load the dataset
            dataset = data_preparation(difficulty=diff)

            indices = random.sample(range(len(dataset)), 30)
            selected_samples = dataset.select(indices)


            true_answers = []
            pred_answers = []
            for i, sample in enumerate(selected_samples):
                question = sample['question']
                true_answers.append(extract_number(sample['answer']))
                print(i, f"True Answer: {true_answers}")
                responses = multi_round_debate(models, tokenizers, question, rounds=2)

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
    parser.add_argument("--rounds", default=2, type=int, help="Total number of rounds in the debate.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

        
