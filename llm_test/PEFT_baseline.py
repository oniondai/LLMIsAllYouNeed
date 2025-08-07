from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

device = "cpu"
model_id = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["q_proj", "v_proj", ...]  # optionally indicate target modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# prints: trainable params: 3,686,400 || all params: 3,089,625,088 || trainable%: 0.1193

# now perform training on your dataset, e.g. using transformers Trainer, then save the model
model.save_pretrained("qwen2.5-3b-lora")
