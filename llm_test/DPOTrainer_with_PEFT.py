from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:100]")  # 只拿 100 条

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["q_proj", "v_proj", ...]  # optionally indicate target modules
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = DPOConfig(
    output_dir="Qwen2.5-0.5B-DPO-PEFT",
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
