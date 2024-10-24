import argparse
import json
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import requests

import anthropic
from dataset_utils.gsm8k import data_preparation, construct_message, construct_init_query

load_dotenv(".env")
anthropic_client = anthropic.Anthropic()


def ollama_chat(model, messages, max_tokens):
    ollama_messages = [
        {
            "role": message["role"], 
            "content": message["content"] if isinstance(message["content"], str) else message["content"][0]["text"]
        }
        for message in messages
    ]

    r = requests.post(
        f"http://localhost:{args.port}/api/chat",
        json={"model": model, "messages": ollama_messages, "keep_alive": -1, "max_tokens": max_tokens, "options": {"num_ctx": 8000}},
    )
    r.raise_for_status()
    output = ""
    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
        if body.get("done", False):
            message["content"] = [
                {
                    "type": "text",
                    "text": output
                }
            ]
            return message


def chat(model, messages, max_tokens):
    if model.startswith("claude"):
        return anthropic_client.beta.prompt_caching.messages.create(model=model, messages=messages, max_tokens=max_tokens)
    else:
        return ollama_chat(model, messages, max_tokens)


def main(args):
    assert args.num_weak <= args.num_agents
    model = ["qwen2-math", "gemma2", "llama3.1"]
    configs = [{"configurable": {"thread_id": f"thread_{n}", "model_id": 0}} for n in range(args.num_agents)]
    for n in range(args.num_weak):
        configs[n]["configurable"]["model_id"] = 1
    print(configs)

    # Prepare dataset
    dataset = data_preparation()

    generated_description = {}
    if not args.restart:
        try:
            generated_description = json.load(open("results/{}/gsm_agents_{}({})_rounds_{}.json".format(args.exp_name, args.num_agents, args.num_weak, args.num_rounds), "r"))
        except:
            generated_description = {}
        dataset = dataset.filter(lambda x: x["question"] not in generated_description.keys())

    print(f"To Do: {dataset.num_rows}")

    for data in tqdm(dataset.__iter__()):
        question = data['question']
        answer = data['answer']
        init_query = construct_init_query(question)
        agent_contexts = [
            [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": init_query,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
            ] for _ in range(args.num_agents)
        ]

        responses = [[] for _ in range(args.num_agents)]
        last_responses = None
        for round in range(args.num_rounds):
            if round != 0:
                for i in range(args.num_agents):
                    agent_contexts[i].append(
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text",
                                    "text": construct_message(last_responses, question=question, idx=i),
                                    "cache_control": {"type": "ephemeral"}
                                }
                            ]
                        }
                    )

            new_responses = []
            for i, config in enumerate(configs):
                output = chat(
                    model=model[config["configurable"]["model_id"]],
                    messages=agent_contexts[i],
                    max_tokens=1024
                )
                output = json.loads(json.dumps(output, default=lambda o: getattr(o, '__dict__', str(o))))
                new_responses.append(output["content"][0]["text"])
                responses[i].append(output["content"][0]["text"])
                agent_contexts[i].append({"role":"assistant", "content": output["content"][0]["text"]})

            last_responses = new_responses
            #accuracy = compute_accuracy(answer, new_responses)
        generated_description[question] = (responses, answer)
        json.dump(generated_description, open("results/{}/gsm_agents_{}({})_rounds_{}.json".format(args.exp_name, args.num_agents, args.num_weak, args.num_rounds), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_weak", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--port", type=str, default="11436")
    args = parser.parse_args()

    main(args)