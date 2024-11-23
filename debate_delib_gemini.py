import google.generativeai as genai
from collections import Counter
import time
import random
import argparse
from datasets import load_dataset

# Configure the Gemini API
genai.configure(api_key="API_KEY)
model = genai.GenerativeModel("gemini-1.0-pro")
random.seed(0)

class ChainOfThoughtWithDeliberation:
    def __init__(self, n):
        self.n = n  # Number of models

    def generate_step(self, current_steps, step_number):
        """
        Generate step proposals using the Gemini API.
        """
        proposals = []
        for i in range(self.n):
            prompt = (
                f"Given the previous steps:\n{current_steps}\n"
                f"What is step {step_number}?"
            )
            response = model.generate_content(prompt)
            proposals.append(response.text.strip())
        for idx, prop in enumerate(proposals):
            print(idx, prop)
        return proposals

    def vote_on_step(self, proposals):
        """
        Conduct voting on the best step using the Gemini API.
        """
        steps = [chr(65 + i) for i in range(len(proposals))]  # Dynamically generate letters A, B, C, ...
        voting_prompt = (
            f"Here are the proposed steps:\n"
            + "\n".join([f"{steps[i]}: {proposals[i]}" for i in range(len(proposals))])
            + "\nReturn ONLY the letter of the step you think is best to arrive at the final answer. If you think multiple choices are good, just randomly pick one of them. Make sure, you ONLY return a letter."
        )
        votes = []
        for i in range(self.n):
            response = model.generate_content(voting_prompt)
            votes.append(response.text.strip().upper())
        vote_counts = Counter(votes)
        most_voted_index = steps.index(max(vote_counts, key=vote_counts.get))
        return proposals[most_voted_index]

    def solve_problem(self, steps):
        """
        Solve the problem using the final steps.
        """
        solutions = []
        for i in range(self.n):
            prompt = (
                f"Given the steps:\n{steps}\n"
                "Solve the problem and provide the answer after #### in the format: #### numerical_answer."
            )
            response = model.generate_content(prompt)
            solutions.append(response.text.strip())
        for i, sol in enumerate(solutions):
            print(i, "'s answer", sol)
        return solutions

    def extract_numerical_answer(self, response):
        """
        Extract numerical answer from the response text.
        """
        try:
            return int(response.split("####")[-1].strip())
        except ValueError:
            return None

    def run_experiment(self, problem):
        """
        Execute the experiment for a given problem.
        """
        current_steps = f"Problem: {problem}"
        final_steps = []

        for step_number in range(1, 5):  # Limit to 3 steps
            print("Sleeping between steps")
            time.sleep(60)  # Sleep to avoid rate limits
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

        print("Starting the experiment using Gemini API...")

        # Load the dataset
        dataset = data_preparation(difficulty=-1)
        indices = random.sample(range(len(dataset)), 650)
        selected_samples = dataset.select(indices)

        true_answers = []
        pred_answers = []

        for i, sample in enumerate(selected_samples):
            question = sample['question']
            print(question)
            print(sample['answer'])
            true_answer = extract_number(sample['answer'])
            true_answers.append(true_answer)
            print(f"{i}, True Answer: {true_answer}")

            experiment = ChainOfThoughtWithDeliberation(n=5)
            results = experiment.run_experiment(question)
            pred_answer = results['final_answer']
            print(f"Predicted Answer: {pred_answer}")
            pred_answers.append(pred_answer)
            if i%10 == 0:
                print(i, "Questions Done")
                print("Accuracy",accuracy(true_answers, pred_answers))
                # Calculate the accuracy
        print("Done")
        print(accuracy(true_answers, pred_answers))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chain of Thought with Deliberation Experiment")
    parser.add_argument('--LLM_num', type=int, default=3, help='Number of LLMs to use')
    parser.add_argument('--rounds', type=int, default=3, help='Number of rounds')
    parser.add_argument('--output_path', type=str, default='results.txt', help='Path to output file')
    args = parser.parse_args()

    main(args)
