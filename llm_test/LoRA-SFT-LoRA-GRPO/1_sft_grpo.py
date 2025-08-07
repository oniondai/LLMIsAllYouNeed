# sft_grpo.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model

model_name = "Qwen/Qwen2-0.5B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 1. 构造 SFT 数据：prompt -> completion
#raw = [
#    {"prompt": "写一句鼓励的话：", "completion": "相信自己，你一定行！"},
#    {"prompt": "翻译：good morning", "completion": "早上好"},
#]
#def fmt(ex):
#    return {"text": f"### 指令\n{ex['prompt']}\n### 回答\n{ex['completion']}"}
#ds = load_dataset("json", data_files={"train": raw}).map(fmt)

from datasets import Dataset

raw = [
    {"prompt": "写一句鼓励的话：", "completion": "相信自己，你一定行！"},
    {"prompt": "翻译：good morning", "completion": "早上好"},
]

ds = Dataset.from_list(raw)          # 直接变成 Dataset，无需写文件
ds = ds.map(lambda ex: {"text": f"### 指令\n{ex['prompt']}\n### 回答\n{ex['completion']}"})

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./sft_lora_grpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    bf16=False,
    fp16=False,
    gradient_checkpointing=False,
    save_steps=50,
    logging_steps=10,
    use_mps_device=(device == "mps"),
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds,
)
trainer.train()
trainer.save_model("./sft_lora_grpo")   # LoRA adapter
