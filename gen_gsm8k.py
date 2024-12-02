import openai
import argparse
import json
import random
import google.generativeai as genai
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# import vllm
SEED = 0
INIT_PROMPT = """Can you solve the following math problem? {} Step-by-step explain your reasoning . Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user",
                "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(
        question)
    return {"role": "user", "content": prefix_string}

def construct_student_message(agents, question, idx):
    prefix_string = "These are the solutions to the problem from your classmates: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the following math problem from the exam? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(
        question)
    return {"role": "user", "content": prefix_string}

def construct_score_message(agents, question, idx):
    prefix_string = "These are the answers to the problem from some agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent answers: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + \
        """
        You are tasked with distributing 100 points among three answers provided by agents based on their accuracy in solving a given math problem. Assign points such that:
        1. The most accurate answer gets the highest points.
        2. Less accurate answers get proportionally fewer points.
        3. Answers with same final results get same points.
        4. The total must always equal 100 points.

        **Output strictly in the following format:**
        \\boxed{{answer 1 points, answer 2 points, answer 3 points}}

        - Explain the points distribution step by step.
        - Do not include any additional formatting such as brackets, labels, or line breaks.
        - Your final response must only consist of the three numerical values in the specified format.

        The original math problem is {}. Provide your distribution plan in the exact format described.
        """.format(question)
    return {"role": "user", "content": prefix_string}

def construct_score_message_v3(agents, question, idx):
    prefix_string = "The original math problem is: {}\n\n".format(question) + \
        "Below are responses from different agents to a math problem:\n\n"

    for agent in agents:
        agent_response = agent[idx]["content"]
        prefix_string += f"Agent {i}'s response: ```{agent_response}```\n\n"

    prefix_string += (
        "You have 100 points to distribute among these answers based on their accuracy in solving the problem. "
        "Your goal is to accurately assess the correctness of each answer step by step and allocate points accordingly.\n\n"
        "**Rules to Follow Strictly:**\n"
        "1. The most accurate answer must get the highest points.\n"
        "2. Less accurate answers should get proportionally fewer points.\n"
        "3. If two or more answers provide the same result, they must receive the same points.\n"
        "4. The total of your points must always equal exactly 100.\n\n"
        "**Instructions:**\n"
        "- First, you must briefly and logically explain your points distribution.\n"
        "- Then, present your final result on a single line in the exact format: \\boxed{{points_for_agent_1, points_for_agent_2, points_for_agent_3}}.\n"
        "- Ensure your final result adheres to the format precisely.\n"
    )

    return {"role": "user", "content": prefix_string}

def construct_score_message_v5(agents, question, idx):
    prefix_string = "The original math problem is: {}\n\n".format(question) + \
        "Below are responses from different agents to a math problem:\n\n"

    for agent in agents:
        agent_response = agent[idx]["content"]
        prefix_string += f"Agent {i}'s response: ```{agent_response}```\n\n"

    prefix_string += (
        "You have 100 points to distribute among these answers based on their accuracy in solving the problem. "
        "Your goal is to accurately assess the correctness of each answer step by step and allocate points accordingly.\n\n"
        "**Rules to Follow Strictly:**\n"
        "1. The most accurate answer must get the highest points.\n"
        "2. Less accurate answers should get proportionally fewer points.\n"
        "3. If two or more answers provide the same result, they must receive the same points.\n"
        "4. The total of your points must always equal exactly 100.\n\n"
        "**Instructions:**\n"
        "- First, you must briefly and logically explain your points distribution.\n"
        "- Then, present your final result on a single line in the exact format: \\boxed{{points_for_agent_1, points_for_agent_2, points_for_agent_3}}.\n"
        "- Ensure your final result adheres to the format precisely.\n"
    )

    return {"role": "user", "content": prefix_string}

def construct_score_message_v4(agents, question, idx):
    prefix_string = (
        "The original math problem is: {}\n\n".format(question) +
        "Below are responses from different agents to the math problem:\n\n"
    )

    for i, agent in enumerate(agents, 1):
        agent_response = agent[idx]["content"]
        prefix_string += f"Agent {i}'s response: ```{agent_response}```\n\n"

    prefix_string += (
        "You have a total of 100 points to distribute among these answers based on their accuracy and quality in solving the problem. "
        "Your goal is to evaluate each response fairly and allocate points proportionally.\n\n"
        "**Evaluation Criteria:**\n"
        "- **Final Answer Accuracy**: Correctness of the final answer.\n"
        "- **Methodology Correctness**: Validity and appropriateness of the steps taken.\n"
        "- **Clarity and Logical Presentation**: How clearly and logically the solution is presented.\n\n"
        "**Rules to Follow Strictly:**\n"
        "1. Distribute the 100 points among the agents based on how well each response meets the evaluation criteria.\n"
        "2. The best answer should receive the highest number of points.\n"
        "3. Less accurate or lower-quality answers should receive proportionally fewer points.\n"
        "4. If two or more answers are equally good, they must receive the same number of points.\n"
        "5. The total points allocated must sum to exactly 100.\n\n"
        "**Instructions:**\n"
        "- **First**, briefly and logically explain the points distribution for each agent's response based on the evaluation criteria.\n"
        "- **Then**, present your final result on a single line in the exact format: \\boxed{points_for_agent_1, points_for_agent_2, points_for_agent_3}.\n"
        "- **Do Not Include** any other text, labels, or formatting outside the explanation.\n"
        "- **Ensure** your final result corresponds to the agents in the order presented and adheres to the format precisely.\n"
    )

    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion):
    return {"role": "model", "content": completion}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def load_model_and_tokenizer(model_name, device_id):
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                        torch_dtype="auto",
                                        attn_implementation="flash_attention_2", 
                                        trust_remote_code=True, 
                                        use_cache=True,
                                        cache_dir="/fs/class-projects/fall2024/cmsc473/c473g001/cache").to(f"{device_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_chat_response(model, tokenizer, agent_context, device):
    try:
        text = tokenizer.apply_chat_template(
            agent_context,
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = tokenizer([text], return_tensors="pt").to(device)
        # print("Input IDs device:", input_ids['input_ids'].device)
        outputs = model.generate(
            **input_ids,
            max_new_tokens=1024,  
            num_beams=5,         
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )

        prompt_length = input_ids['input_ids'].shape[1]
        generated_ids = outputs[0][prompt_length:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    except Exception as e:
        completion = "API FAILED"
        print(f"From agent {i}, round {round}: {completion} \nError: {str(e)}")
    return completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple argparse example")
    parser.add_argument("num_models", type=int, default=1, help="Number of models")
    parser.add_argument("agents", type=int, default=1, help="Number of agents")
    parser.add_argument("rounds", type=int, default=1, help="Number of rounds")
    parser.add_argument("--selfref", action="store_true", help="Include self-reference (default: False)")
    parser.add_argument("--history", action="store_true", help="Include chat history (default: False)")


    args = parser.parse_args()

    agents = args.num_models if args.num_models > 1 else args.agents
    rounds = args.rounds
    if args.selfref:
        print("SELF REFERENCE")
    random.seed(SEED)
    # genai.configure(api_key="")
    # model = genai.GenerativeModel("gemini-1.0-pro")

    # FAMILY = "Qwen"
    # FAMILY = "meta-llama"
    # FAMILY = "mistralai"
    # FAMILY = "microsoft"
    # mistralai/Ministral-8B-Instruct-2410
    # microsoft/Phi-3-small-8k-instruct 8b
    # MODEL = FAMILY + "/" + "Qwen2.5-7B-Instruct"
    # MODEL = FAMILY + "/" + "Llama-3.1-8B-Instruct"
    # MODEL = FAMILY + "/" + "Phi-3-small-8k-instruct"
    
    model_names = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Ministral-8B-Instruct-2410", "microsoft/Phi-3-small-8k-instruct"][:args.num_models]
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"][:args.num_models]
    models_tokenizers = [load_model_and_tokenizer(model_name, device) for model_name, device in zip(model_names, devices)]

    generated_description = {}

    questions = read_jsonl("data/test.json")
    random.shuffle(questions)
    # request_count = 0
    for data in tqdm(questions[:650]):
        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user",
                            "content": INIT_PROMPT.format(
                                question)}] for _ in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    if args.selfref:
                        agent_contexts_other = agent_contexts
                    else:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1:]
                    # message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    message = construct_score_message_v5(agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)

                model, tokenizer = models_tokenizers[i]
                # print("Model device:", next(model.parameters()).device)
                if i > 1: # phi have different conversation schema
                    for message in agent_context:
                        if message["role"] == "model":
                            message["role"] = "assistant"
                if args.history:
                    completion = generate_chat_response(model, tokenizer, agent_context=agent_context, device=devices[i])
                else:
                    completion = generate_chat_response(model, tokenizer, agent_context=[agent_context[-1]], device=devices[i])

                assistant_message = construct_assistant_message(completion=completion)
                agent_context.append(assistant_message)
                # print(agent_context)


        generated_description[question] = (agent_contexts, answer)
        # break
    # print(generated_description)
    json.dump(generated_description, open("gsm_multi_score_v5_{}_{}_CoT_650.json".format(agents, rounds), "w"))

