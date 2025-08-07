from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:100]")  # 只拿 100 条

#training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO")
#training_args = DPOConfig(
#    output_dir="Qwen2.5-0.5B-DPO",
#    bf16=False,          # 关 bfloat16
#    fp16=False,          # 也关 fp16，Mac 用不到
#    use_mps_device=True  # Apple Silicon 用 mps；如果是 Intel Mac 就删掉这一行
#)
training_args = DPOConfig(
    output_dir="Qwen2.5-0.5B-DPO",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    bf16=False,
    fp16=False,
    use_mps_device=True,
    gradient_checkpointing=True,   # 激活检查点，省 30-40% 内存
    max_length=512,                # 再砍序列长度
    remove_unused_columns=False,
    logging_steps=10,
)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)
trainer.train()
