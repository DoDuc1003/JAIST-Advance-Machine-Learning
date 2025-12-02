import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
import huggingface_hub

huggingface_hub.login("hf_token")

import wandb
wandb.login(key="wandb_token")

system_message = """
You are a Vision-Language Model specialized in reasoning over synthetic 3D scenes from the CLEVR dataset.

Scene rules:
- Each image contains 3–10 objects.
- Objects have exactly four attributes:
  • Shape: cube, sphere, cylinder
  • Color: gray, red, blue, brown, green, purple, cyan, yellow
  • Size: small, large
  • Material: metal, rubber
- No other attributes exist.
- Objects do not intersect, are fully or partially visible, and appear on a clean, uncluttered background.

Reasoning rules:
- Always reason strictly based on the visible objects in the scene.
- For “same” queries, compare identical attributes across objects.
- For “other” queries, exclude the reference object.
- For existence queries (“are there…”, “any”), answer yes/no based on the presence of at least one valid object.
- For counting queries, return the exact number as a token.
- If no valid matching object exists, answer “no”.
- Never hallucinate objects or attributes.

Answer vocabulary:
You must ONLY use tokens from this fixed list:
"0","gray","cube","purple","yes","small","brown","red","blue",
"7","5","8","metal","6","rubber","1","sphere","cylinder",
"3","10","2","yellow","cyan","green","9","large","no","4".

Formatting rules (strict):
Your response MUST ALWAYS follow this exact structure:
"The answer is: <answer> </answer>.
Because: <explain> <explanation> </explain>"

Where:
- <answer> is exactly one token from the allowed list.
- <explanation> is a short, factual justification (max 15 words).
- Do not add extra text, punctuation, newlines, or reasoning outside the explanation tag.

Explanation guidelines:
- Keep explanations short and simple.
- Use typical CLEVR-style phrases such as:
  • “because I see …”
  • “because there is …”
  • “because no object matches …”
  • “because the sphere is …”
- Do not introduce new concepts beyond CLEVR attributes.
- For existence queries (e.g., "Are there...?", "Is there...?"), the explanation must be directly linked to the visual presence of objects:
  - If the object is present: "because there is a <object>."
  - If the object is not present: "because no <object> is visible."
  - For counting queries (e.g., "How many..."): Return the exact count as a token, explaining the count concisely.
  - Example: "Are there any cylinders?" 
    - **Answer**: "The answer is: yes </answer>. Because: because there is a cylinder. </explanation>"
  - Example: "Is there a red sphere?"
    - **Answer**: "The answer is: yes </answer>. Because: because there is a red sphere. </explanation>"
"""


from datasets import load_dataset

dataset_id = "Kudod/ExVQA-AML"
all_dataset = load_dataset(dataset_id)
all_dataset = [sample for sample in all_dataset['train']]


import random
# Trộn dữ liệu ngẫu nhiên
random.shuffle(all_dataset)

# Tính chỉ số phân chia
split_index = int(0.8 * len(all_dataset))

# Chia dữ liệu thành train và eval
train_data = all_dataset[:split_index]
eval_data = all_dataset[split_index:]

# Kiểm tra kích thước của các tập dữ liệu
print(f"Training data size: {len(train_data)}")
print(f"Evaluation data size: {len(eval_data)}")

# In một số mẫu của tập huấn luyện và kiểm tra
print(f"Sample from train data: {train_data[:1]}")
print(f"Sample from eval data: {eval_data[:1]}")

CLASSES = [
    "0", "gray", "cube", "purple", "yes", "small", "brown", "red",
    "blue", "7", "5", "8", "metal", "6", "rubber", "1", "sphere",
    "cylinder", "3", "10", "2", "yellow", "cyan", "green", "9",
    "large", "no", "4",
]

import ast

def format_data(sample):
    explanation = sample["explanation"]
    choosen = []
    choosen_all = []
    explanation_list = ast.literal_eval(explanation)
    random.seed(1003)
    if isinstance(explanation_list, list):
        print("Explanation is list")
        choosen_all.append(random.choice(explanation_list))
        # for item in explanation_list:
        #     if (item.split()[0] not in choosen):
        #         choosen.append(item.split()[0])
        #         choosen_all.append(item)
    
    # Lấy câu trả lời từ 'label', nếu không có trong CLASSES thì ép buộc về một giá trị hợp lệ
    answer = sample["label"]
    if answer not in CLASSES:
        answer = "unknown"  # Hoặc có thể là một giá trị mặc định nếu không có trong CLASSES
        print(sample)

    # Tạo mẫu cho mỗi câu giải thích khác nhau
    formatted_samples = []
    for exp in choosen_all:
        explanation_text = exp

        # Tạo mẫu đã được định dạng
        formatted_sample = {
            "images": [sample["image"]],
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_message
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": sample["image"],
                        },
                        {
                            "type": "text",
                            "text": f"{sample['query']} and explain",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The answer is: <answer> {answer} </answer>. \n Because: <explain> {explanation_text} </explain>"
                        }
                    ],
                },
            ]
        }

        # Thêm mẫu vào danh sách kết quả
        formatted_samples.append(formatted_sample)
        break

    return formatted_samples


# Tạo dữ liệu cho tập huấn luyện và kiểm tra với format_data
train_data_formatted = []
eval_data_formatted = []

# Duyệt qua tất cả các mẫu trong train_data
for sample in train_data:
    # Lặp qua tất cả các giải thích có từ đầu tiên khác nhau
    formatted_samples = format_data(sample)
    for item in formatted_samples:
        train_data_formatted.append(item)
    # train_data_formatted.extend(formatted_samples)

# Duyệt qua tất cả các mẫu trong eval_data
for sample in eval_data:
    # Lặp qua tất cả các giải thích có từ đầu tiên khác nhau
    formatted_samples = format_data(sample)
    for item in formatted_samples:
        eval_data_formatted.append(item)
    # eval_data_formatted.extend(formatted_samples)

# Kiểm tra kích thước của dữ liệu đã định dạng
print(f"Formatted training data size: {len(train_data_formatted)}")
print(f"Formatted evaluation data size: {len(eval_data_formatted)}")

# In một số mẫu của dữ liệu đã định dạng
print(f"Sample from formatted train data: {train_data_formatted[:1]}")
print(f"Sample from formatted eval data: {eval_data_formatted[:1]}")

# In một số mẫu của dữ liệu đã định dạng
print(f"Sample from formatted train data: {train_data_formatted[-1]}")
print(f"Sample from formatted eval data: {eval_data_formatted[-1]}")

# Kiểm tra kích thước của dữ liệu đã định dạng
print(f"Formatted training data size: {len(train_data_formatted)}")
print(f"Formatted evaluation data size: {len(eval_data_formatted)}")

import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
# model_id = "Qwen/Qwen3-VL-4B-Instruct"
model_id = "Qwen/Qwen3-VL-2B-Instruct"
# model_id = "Qwen/Qwen2-VL-2B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

from transformers import BitsAndBytesConfig

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_id,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     quantization_config=bnb_config
# )
# processor = Qwen2VLProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id)

from peft import LoraConfig

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.05,
    r=32,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

from trl import SFTConfig

# Configure training arguments
training_args = SFTConfig(
    output_dir="qwen2_5-3b-instruct-exvqa-random=explain",  # Directory to save the model
    num_train_epochs=10,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    max_length=None,
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    # optim="adamw_torch",
    learning_rate=5e-5,  # Learning rate for training
    # betas=(0.9, 0.98),
    # lr_scheduler_type="linear",
    # Logging and evaluation
    logging_steps=30,  # Steps interval for logging
    eval_steps=30,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=300,  # Steps interval for saving
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=True,  # Whether to push model to Hugging Face Hub
    report_to="wandb",  # Reporting tool for tracking metrics
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data_formatted,
    eval_dataset=eval_data_formatted,
    peft_config=peft_config,
    processing_class=processor,
)
trainer.train()

trainer.save_model(training_args.output_dir)
trainer.push_to_hub()