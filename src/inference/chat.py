"""
CodeGuide-LLM 推理 Demo（终端交互）
用法: python src/inference/chat.py --model models/final/codeguide-7b
"""
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SYSTEM = """你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手。
当用户提出算法题时，请逐步讲解解题思路，引导用户自己思考，而不是直接给出完整答案。"""


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages: list, max_new_tokens: int = 1024) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/final/codeguide-7b")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    print(f"加载模型：{args.model} ...")
    model, tokenizer = load_model(args.model)
    print("模型就绪。输入 'quit' 退出。\n")

    history = [{"role": "system", "content": SYSTEM}]

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        response = generate(model, tokenizer, history, args.max_new_tokens)
        history.append({"role": "assistant", "content": response})
        print(f"\nCodeGuide:\n{response}\n")


if __name__ == "__main__":
    main()
