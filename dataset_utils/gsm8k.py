from datasets import load_dataset


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


def construct_init_query(question):
    return "Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. ".format(question)


def construct_message(last_reponses, question, idx):
    if last_reponses is None or len(last_reponses) <= 1:
        return "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."

    prefix_string = "These are the solutions to the problem from other agents: "

    for i, last_response in enumerate(last_reponses):
        if i == idx:
            continue
        agent_response = last_response#["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return prefix_string