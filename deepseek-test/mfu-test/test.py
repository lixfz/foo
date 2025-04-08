from datasets import load_from_disk
from transformers import AutoTokenizer

# print("loading datasets")
# lm_datasets = load_from_disk("/slurmhome/aps/data/llama2/continue_train/zh/wiki_zh_1g_dataset_0.29b", keep_in_memory=False)['train']
# print(lm_datasets)
# import json
# with open('a.txt', 'w') as f:
#     json.dump(lm_datasets[0], f)
# print(lm_datasets[0])
print("===========================")

# lm_datasets = lm_datasets.train_test_split(
#         test_size=0.2, seed=1024, shuffle=True
#     )
# print(lm_datasets)

# print(len(lm_datasets[0]['input_ids']))

# tokenizer = AutoTokenizer.from_pretrained(
#         "/slurmhome/aps/lumk/test_network/llama2_chinese_merged_tokenizer", trust_remote_code=True
#     )
# print(tokenizer)
# 60708, 4096

# 'input_ids', 'attention_mask', 'labels'
from datasets import Dataset
import pandas as pd
import numpy as np
import json

with open('sample_dta.json', 'r') as f:
    one_row = json.load(f)

input_ids = np.repeat([one_row['input_ids']], 100, axis=0)
attention_mask = np.repeat([one_row['attention_mask']], 100, axis=0)
labels = np.repeat([one_row['labels']], 100, axis=0)
data = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels
}
dataset = Dataset.from_dict(data)
print(dataset)
print(dataset[0])
