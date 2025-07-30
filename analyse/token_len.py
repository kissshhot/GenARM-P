
import json
import torch
from transformers import LlamaTokenizer, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 替换为你的 JSONL 文件路径
jsonl_file_path = '/home/dyf/rl/GenARM/data/full-hh-rlhf.jsonl'
# 替换为你的 LLaMA 模型的 tokenizer 路径或名称
tokenizer_name = '/data1/dyf/model/llama-7b-sft-float32/'

# 加载 LLaMA 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# 初始化列表来存储长度
chosen_lengths = []
rejected_lengths = []
length_diffs = []

# 读取 JSONL 文件
with open(jsonl_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        chosen_text = data.get('chosen', '')
        rejected_text = data.get('rejected', '')

        # 使用 tokenizer 编码文本并获取长度
        chosen_length = len(tokenizer.encode(chosen_text))
        rejected_length = len(tokenizer.encode(rejected_text))

        chosen_lengths.append(chosen_length)
        rejected_lengths.append(rejected_length)
        length_diffs.append(abs(chosen_length - rejected_length))

# 转换为 NumPy 数组方便计算分位数
chosen_lengths_np = np.array(chosen_lengths)
rejected_lengths_np = np.array(rejected_lengths)
length_diffs_np = np.array(length_diffs)

# 计算分位数
quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
chosen_quantiles = np.quantile(chosen_lengths_np, quantiles)
rejected_quantiles = np.quantile(rejected_lengths_np, quantiles)
length_diffs_quantiles = np.quantile(length_diffs_np, quantiles)

# 输出结果
print("Chosen Lengths Quantiles:")
for q, val in zip(quantiles, chosen_quantiles):
    print(f"{q*100:.0f}%: {val}")

print("\nRejected Lengths Quantiles:")
for q, val in zip(quantiles, rejected_quantiles):
    print(f"{q*100:.0f}%: {val}")

print("\nLength Differences Quantiles:")
for q, val in zip(quantiles, length_diffs_quantiles):
    print(f"{q*100:.0f}%: {val}")

# 可视化
plt.figure(figsize=(15, 5))

# Chosen Lengths Distribution
plt.subplot(1, 3, 1)
sns.histplot(chosen_lengths_np, kde=True, bins=50, color='blue')
plt.title('Chosen Lengths Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')

# Rejected Lengths Distribution
plt.subplot(1, 3, 2)
sns.histplot(rejected_lengths_np, kde=True, bins=50, color='red')
plt.title('Rejected Lengths Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')

# Length Differences Distribution
plt.subplot(1, 3, 3)
sns.histplot(length_diffs_np, kde=True, bins=50, color='green')
plt.title('Length Differences Distribution')
plt.xlabel('Length Difference')
plt.ylabel('Frequency')

plt.tight_layout()
# plt.show()
plt.savefig("/home/dyf/rl/GenARM/analyse/token_len.jpg")