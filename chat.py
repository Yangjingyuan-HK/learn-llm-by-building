"""
交互式对话

用法:
  python chat.py          # Pretrain 模式
  python chat.py --sft    # SFT 模式
"""
import os
import sys

import torch
from transformers import AutoTokenizer

from llm import Config, TransformerLM

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    use_sft = "--sft" in sys.argv
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer"))

    path = os.path.join(ROOT, "checkpoints", f"model_{'sft' if use_sft else 'pretrain'}.pth")

    ckpt = torch.load(path, map_location=DEVICE)
    cfg = Config()
    for k, v in ckpt["cfg"].items():
        setattr(cfg, k, v)
    model = TransformerLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    temp = 0.8
    max_new = 128
    while True:
        try:
            text = input("Input ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text == ":q":
            break
        if text.startswith(":temp "):
            temp = float(text[6:])
            continue
        if text.startswith(":len "):
            max_new = int(text[5:])
            continue

        if use_sft:
            msgs = [{"role": "user", "content": text}]
            prompt = tokenizer.encode(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)
        else:
            prompt = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False)

        ids = torch.tensor(prompt, device=DEVICE).unsqueeze(0)
        out = model.generate(ids, max_new, temp)
        reply = tokenizer.decode(out[0, len(prompt):], skip_special_tokens=True)
        print(f"AI ▶ {reply}\n")

    print("再见!")


if __name__ == "__main__":
    main()