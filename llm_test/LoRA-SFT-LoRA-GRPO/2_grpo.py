# grpo.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel

model_name = "Qwen/Qwen2-0.5B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 1. 加载基础模型 + SFT-LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = PeftModel.from_pretrained(base_model, "./sft_lora_grpo", is_trainable=True)

# 2. 构造 GRPO 数据（prompt 列必须有）
ds = load_dataset("trl-lib/tldr", split="train[:100]")  # 先用 100 条测试

# 3. rule-based reward：答案里含「the」给 +1，否则 0
def reward_has_the(completions, **kwargs):
    return [1.0 if " the " in c.lower() else 0.0 for c in completions]

# 4. GRPO 训练参数
training_args = GRPOConfig(
    output_dir="./grpo_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    bf16=False,
    fp16=False,
    gradient_checkpointing=True,
    max_length=512,
    logging_steps=5,
    use_mps_device=(device == "mps"),
)

trainer = GRPOTrainer(
    model=model,                # 已含 LoRA adapter
    reward_funcs=reward_has_the,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./grpo_lora")   # 继续保存新的 LoRA adapter
