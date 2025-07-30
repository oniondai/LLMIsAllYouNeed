from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 加载MMLU中与金融相关的数据集（例如计量经济学）
# 可选的金融相关子集：'econometrics', 'high_school_macroeconomics', 'high_school_microeconomics', 'professional_accounting'
dataset = load_dataset("cais/mmlu", "econometrics")["test"]  # 替换为金融相关子集
sample_data = dataset.select(range(100))  # 取前100个样本

# 2. 加载模型（替换为你的模型路径）
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
cache_dir="/Users/daicong/Documents/5.model_weight_file/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def test_llm():
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


def test_finance_data():
    # 3. 定义Prompt模板和生成函数
    prompt_template = """
    请基于金融专业知识，从以下选项中选择正确答案。只需输出选项字母（如A、B、C、D），无需额外解释。

    问题：{question}
    选项：
    A. {option_a}
    B. {option_b}
    C. {option_c}
    D. {option_d}

    答案：
    """

    def generate_answer(question, options):
        prompt = prompt_template.format(
            question=question,
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3]
        )

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512
        )
        answer = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
        return answer

    # 4. 测评与结果分析
    correct = 0
    total = 0
    results = []

    for item in sample_data:
        question = item["question"]
        options = item["choices"]
        true_answer = item["answer"]
        true_letter = chr(65 + true_answer)  # 索引转A/B/C/D
        
        pred_letter = generate_answer(question, options)
        correct += (pred_letter == true_letter)
        total += 1
        results.append({
            "question": question,
            "options": options,
            "true": true_letter,
            "pred": pred_letter,
            "correct": pred_letter == true_letter
        })

    print(f"测评准确率：{correct / total:.2f} ({correct}/{total})")

    # 保存错误样本
    wrong_samples = [r for r in results if not r["correct"]]
    with open("finance_wrong_samples.txt", "w", encoding="utf-8") as f:
        for idx, sample in enumerate(wrong_samples):
            f.write(f"样本{idx+1}：\n")
            f.write(f"问题：{sample['question']}\n")
            f.write(f"选项：{sample['options']}\n")
            f.write(f"正确答案：{sample['true']}，模型答案：{sample['pred']}\n\n")
    
if __name__ == "__main__":
    # test_llm()
    test_finance_data() # 测评准确率：0.29 (29/100)