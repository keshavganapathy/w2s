from datasets import load_dataset
import json
from datasets.download.download_config import DownloadConfig
config = DownloadConfig(max_retries=3)
dataset = load_dataset(
        "furonghuang-lab/Easy2Hard-Bench",
        "E2H-GSM8K",
        split="eval",
        download_config=config
    ).select_columns(
        ["question", "answer", "rating_quantile"]
    ).sort(
        "rating_quantile"
    )
print(dataset)
# save to json file

split = "e2h_test"
dataset.to_json(split + '.json', orient='records', lines=True)
generated_description = {}
for data in dataset:
    generated_description[data["question"]] = (data["answer"], data["rating_quantile"])
json.dump(generated_description, open("e2h_test_sorted.json", "w"))