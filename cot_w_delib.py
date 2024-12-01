import random
import time
import argparse
from collections import Counter
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(0)

auth_token = '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'

def load_model_and_tokenizer(model_name, device_id):
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                        torch_dtype="auto",
                                        attn_implementation="flash_attention_2", 
                                        trust_remote_code=True, 
                                        use_cache=True,
                                        token=auth_token,
                                        cache_dir="/fs/class-projects/fall2024/cmsc473/c473g001/cache").to(f"{device_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=auth_token)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    output_tokens = outputs[0][input_length:]  # Get the generated tokens
    generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return generated_text.strip()

class ChainOfThoughtWithDeliberation:
    def __init__(self, models_tokenizers_devices):
        self.models_tokenizers = models_tokenizers_devices
        self.n = len(models_tokenizers_devices)

    def generate_step(self, current_steps, step_number):
        """
        Generate step proposals using the models.
        """
        proposals = []
        for i, (model, tokenizer, device) in enumerate(self.models_tokenizers):
            prompt = (
                f"Given the previous steps:\n{current_steps}\n"
                f"What is step {step_number}?"
            )
            response = generate_response(model, tokenizer, prompt, device)
            proposals.append(response.strip())
        for idx, prop in enumerate(proposals):
            print(idx, prop)
        return proposals

    def vote_on_step(self, proposals):
        """
        Conduct voting on the best step using the models.
        """
        steps = [chr(65 + i) for i in range(len(proposals))]  # Dynamically generate letters A, B, C, ...
        voting_prompt = (
            f"Here are the proposed steps:\n"
            + "\n".join([f"{steps[i]}: {proposals[i]}" for i in range(len(proposals))])
            + "\nReturn ONLY the letter of the step you think is best to arrive at the final answer. If you think multiple choices are good, just randomly pick one of them. Make sure, you ONLY return a letter."
        )
        votes = []
        for i, (model, tokenizer, device) in enumerate(self.models_tokenizers):
            response = generate_response(model, tokenizer, voting_prompt, device)
            votes.append(response.strip().upper())
        vote_counts = Counter(votes)
        most_voted_index = steps.index(max(vote_counts, key=vote_counts.get))
        return proposals[most_voted_index]

    def solve_problem(self, steps):
        """
        Solve the problem using the final steps.
        """
        solutions = []
        for i, (model, tokenizer, device) in enumerate(self.models_tokenizers):
            prompt = (
                f"Given the steps:\n{steps}\n"
                "Solve the problem and provide the answer after #### in the format: #### numerical_answer."
            )
            response = generate_response(model, tokenizer, prompt, device)
            solutions.append(response.strip())
        for i, sol in enumerate(solutions):
            print(i, "'s answer", sol)
        return solutions

    def extract_numerical_answer(self, response):
        try:
            # Split the response by spaces, iterate backwards to find the last float
            for part in reversed(response.split()):
                try:
                    return float(part)
                except ValueError:
                    continue
            return None  # Return None if no float is found
        except Exception as e:
            return None

    def run_experiment(self, problem):
        """
        Execute the experiment for a given problem.
        """
        current_steps = f"Problem: {problem}"
        final_steps = []

        for step_number in range(1, 4):  # Limit to 3 steps
            print("Sleeping between steps")
            time.sleep(60)  # Sleep to avoid overload
            proposals = self.generate_step(current_steps, step_number)
            selected_step = self.vote_on_step(proposals)
            print("Selected", step_number, "is", selected_step)
            final_steps.append(selected_step)
            current_steps += f"\nStep {step_number}: {selected_step}"

        # Solve the problem
        solutions = self.solve_problem("\n".join(final_steps))
        numerical_answers = [self.extract_numerical_answer(sol) for sol in solutions if sol]
        print("\nFinal Answers")
        for i,num in enumerate(numerical_answers):
            print(i, num)
        final_answer = Counter(numerical_answers).most_common(1)[0][0]

        return {
            "final_steps": final_steps,
            "solutions": solutions,
            "final_answer": final_answer
        }

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

def extract_number(text):
    """Extract numerical answer from text"""
    """
    Extract numerical answer from the response text.
    """
    try:
        return float(text.split("####")[-1].strip())
    except ValueError:
        return None

def best_of_n(responses):
    """
    Choose the most common answer from the list of responses.
    """
    answers = [response for response in responses if response is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]

def accuracy(true_answers, pred_answers):
    """
    Compute the accuracy between true and predicted answers.
    """
    correct = sum(t == p for t, p in zip(true_answers, pred_answers))
    return correct / len(true_answers)

def main(args):
    with open(args.output_path, "a") as file:
        file.write(f"Number of LLMs: {args.LLM_num}, Number of rounds: {args.rounds}\n")

        print("Starting the experiment using local models...")

        # Load the dataset
        dataset = data_preparation(difficulty=-1)
        indices = random.sample(range(len(dataset)), 650)
        selected_samples = dataset.select(indices)

        true_answers = []
        pred_answers = []

        # Load the models
        model_names = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Ministral-8B-Instruct-2410"]
        devices = ["cuda:0", "cuda:1", "cuda:2"]
        models_tokenizers_devices = []
        for model_name, device in zip(model_names, devices):
            model, tokenizer = load_model_and_tokenizer(model_name, device)
            models_tokenizers_devices.append( (model, tokenizer, device) )

        for i, sample in enumerate(selected_samples):
            question = sample['question']
            print(question)
            print(sample['answer'])
            true_answer = extract_number(sample['answer'])
            true_answers.append(true_answer)
            print(f"{i}, True Answer: {true_answer}")

            experiment = ChainOfThoughtWithDeliberation(models_tokenizers_devices)
            results = experiment.run_experiment(question)
            pred_answer = results['final_answer']
            print(f"Predicted Answer: {pred_answer}")
            pred_answers.append(pred_answer)
            if i%10 == 0:
                print(i, "Questions Done")
                print("Accuracy",accuracy(true_answers, pred_answers))
        print("Done")
        print(accuracy(true_answers, pred_answers))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chain of Thought with Deliberation Experiment")
    parser.add_argument('--LLM_num', type=int, default=3, help='Number of LLMs to use')
    parser.add_argument('--rounds', type=int, default=3, help='Number of rounds')
    parser.add_argument('--output_path', type=str, default='results.txt', help='Path to output file')
    args = parser.parse_args()

    main(args)
