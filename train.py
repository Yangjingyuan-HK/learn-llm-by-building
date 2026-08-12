"""
两阶段训练: Pretrain → 复制权重 → SFT

用法:
  python train.py

数据准备: 从 ModelScope 下载到 dataset/
  dataset/pretrain_t2t_mini.jsonl  (1.2GB)
  dataset/sft_t2t_mini.jsonl       (1.6GB)
"""
import copy
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

from llm import Config, TransformerLM
from dataloader import PretrainDataset, SFTDataset, collate_fn

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.path.join(ROOT, "checkpoints")

# 超参
PRETRAIN_EPOCHS, PRETRAIN_BATCH, PRETRAIN_SEQ = 3, 32, 512
SFT_EPOCHS, SFT_BATCH, SFT_SEQ = 3, 16, 512
WARMUP, LR = 4000, 1e-4
LOG_EVERY = 50


# 训练循环
def train(stage, model, cfg, dataset, batch_size, num_epochs):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    collate_fn=collate_fn, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min((s+1)**-0.5, (s+1)*WARMUP**-1.5) * cfg.d_model**-0.5 / LR)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    model.to(DEVICE).train()

    print(f"[{stage}] epochs={num_epochs} batch={batch_size} device={DEVICE}")
    global_step = 0
    for epoch in range(num_epochs):
        t0, running = time.time(), 0.0
        for input_ids, labels in dl:
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast("cuda"):
                    _, loss = model(input_ids, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(input_ids, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running += loss.item()
            global_step += 1
            if global_step % LOG_EVERY == 0:
                print(f"  [{stage}] ep{epoch} step{global_step} "
                      f"loss={running/LOG_EVERY:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                      f"{(time.time()-t0):.0f}s")
                running, t0 = 0.0, time.time()
        print(f"  [{stage}] epoch {epoch} done")
    return model


# 主流程
def main():
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer"))
    cfg = Config()
    cfg.vocab_size = len(tokenizer)
    print(f"vocab={cfg.vocab_size} device={DEVICE} "
          f"params={sum(p.numel() for p in TransformerLM(cfg).parameters()):,}")

    ds_dir = os.path.join(ROOT, "dataset")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Pretrain
    pre_model = train("Pretrain", TransformerLM(cfg), cfg,
                      PretrainDataset(os.path.join(ds_dir, "pretrain_t2t_mini.jsonl"),
                                      tokenizer, PRETRAIN_SEQ),
                      PRETRAIN_BATCH, PRETRAIN_EPOCHS)
    torch.save({"state_dict": pre_model.state_dict(), "cfg": cfg.__dict__},
               os.path.join(CKPT_DIR, "model_pretrain.pth"))
    print("[Pretrain] 已保存 → checkpoints/model_pretrain.pth")

    # SFT (复制 Pretrain 权重为起点)
    sft_model = TransformerLM(cfg)
    sft_model.load_state_dict(copy.deepcopy(pre_model.state_dict()))
    sft_model = train("SFT", sft_model, cfg,
                      SFTDataset(os.path.join(ds_dir, "sft_t2t_mini.jsonl"),
                                 tokenizer, SFT_SEQ),
                      SFT_BATCH, SFT_EPOCHS)
    torch.save({"state_dict": sft_model.state_dict(), "cfg": cfg.__dict__},
               os.path.join(CKPT_DIR, "model_sft.pth"))
    print("[SFT] 已保存 → checkpoints/model_sft.pth")
    print("\n训练完成! 用 python chat.py 或 python chat.py --sft 对话")


if __name__ == "__main__":
    main()