import json
import numpy as np
import re
from tqdm import tqdm

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution


def compute_accuracy(gt, pred_solution):
    answers = solve_math_problems(gt)

    if answers is None:
        return None

    if type(pred_solution) == list:
        pred_answers = []

        for pred in pred_solution:
            pred_answer = parse_answer(pred)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred)

            pred_answers.append(pred_answer)

        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_answer(pred_solution)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

    if pred_answer is None:
        return 1

    if float(answers) == float(pred_answer):
        return 1
    else:
        return 0


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

if __name__ == "__main__":
    response_dict = json.load(open("data/gsm_3_3.json", "r"))

    questions = list(response_dict.keys())

    accuracies = []

    for question in tqdm(questions):
        responses, gt = response_dict[question]

        pred_solutions = []
        for response in responses:
            pred_solution = response[3]['parts']

            pred_solutions.append(pred_solution)

        accurate = compute_accuracy(gt, pred_solutions)

        if accurate is not None:
            accuracies.append(float(accurate))

        print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
