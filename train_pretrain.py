import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')

'''
预训练(Pretrain):
LLM首先要学习的并非直接与人交流，而是让网络参数中充满知识的墨水，“墨水” 理论上喝的越饱越好，
产生大量的对世界的知识积累。 预训练就是让Model先埋头苦学大量基本的知识，
例如从Wiki百科、新闻、书籍整理大规模的高质量训练数据。 
这个过程是“无监督”的，即人类不需要在过程中做任何“有监督”的校正，而是由模型自己从大量文本中总结规律学习知识点。 
模型此阶段目的只有一个：学会词语接龙。例如我们输入“秦始皇”四个字，它可以接龙“是中国的第一位皇帝”。
'''

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr):
    # CosineAnnealingLR是PyTorch提供的一个余弦退火学习率调度器，可以在训练过程中对学习率进行余弦退火调度。
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, wandb):
    # 交叉熵损失函数
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 计算model的损失，包括主损失和辅助损失，并进行适当的归一化和调整，方便后面的梯度更新操作
        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps
        # 对损失进行缩放并反向传播梯度。缩放损失的目的是为了避免在使用半精度浮点数（FP16）时出现数值下溢的问题。
        scaler.scale(loss).backward()
        
        # 检查当前步骤是否是累积步数的整数倍
        if (step + 1) % args.accumulation_steps == 0:
            # 优化器的梯度进行反缩放。这一步是为了确保梯度在更新参数之前被正确缩放。
            scaler.unscale_(optimizer)
            # 对梯度进行裁剪，以防止梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 更新优化器的参数。如果在反缩放过程中检测到无穷大或 NaN 值，优化器的参数更新将被跳过。
            scaler.step(optimizer)
            # 更新缩放器的比例因子。这一步是为了确保在下一次迭代中，梯度的缩放比例是正确的。
            scaler.update()
            # 清空优化器的梯度。这一步是为了确保在下一次迭代中，梯度不会被重复使用。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'
            # 保存模型的参数
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

# 初始化 nccl分布式集群训练
def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true") # 是否进行集群部署
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1) # 是否单机部署
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # DistributedSampler是PyTorch提供的一个分布式采样器，可以在分布式训练中对数据进行分布式采样。
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # DataLoader是PyTorch提供的一个数据加载器，可以在训练过程中对数据进行批量加载。
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    
    # AMP（Automatic Mixed Precision）是 自动混合精度 计算技术，可以在保持模型精度的同时提高训练速度。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # AdamW是Adam的一个变种，主要是为了解决Adam优化器的权重衰减问题。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
