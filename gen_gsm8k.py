import openai
import json
import random
import google.generativeai as genai
import time
from tqdm import tqdm


def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user",
                "parts": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["parts"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(
        question)
    return {"role": "user", "parts": prefix_string}


def construct_assistant_message(completion):
    return {"role": "model", "parts": completion}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


if __name__ == "__main__":
    agents = 3
    rounds = 3
    random.seed(0)

    genai.configure(api_key=)
    model = genai.GenerativeModel("gemini-1.0-pro")

    generated_description = {}

    questions = read_jsonl("data/test.json")
    # random.shuffle(questions)
    request_count = 0
    for data in tqdm(questions):
        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user",
                            "parts": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(
                                question)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1:]
                    message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)

                try:
                    chat = model.start_chat(
                        history=agent_context[:-1]
                    )
                    response = chat.send_message(agent_context[-1]["parts"])
                    request_count += 1
                    if request_count == 15:
                        request_count = 0
                        time.sleep(60)
                    completion = response.text
                except:
                    completion = "API FAILED"
                    print(f"From agent {i}, round {round}:" + completion)
                assistant_message = construct_assistant_message(completion=completion)
                agent_context.append(assistant_message)
                # print(agent_context)

        generated_description[question] = (agent_contexts, answer)
        break

    json.dump(generated_description, open("gsm_{}_{}.json".format(agents, rounds), "w"))

