import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
from dataclasses import dataclass
from transformers import BertTokenizer

torch.manual_seed(1334)


Encoder = BertTokenizer.from_pretrained("src/token/Encoder")
Encoder.model_max_length = 1e8
# TokenIds
print("Total TokenIds", len(Encoder))


@dataclass
class GPTConfig:
    block_size: int = 512  # context_length, max_seq 文本最大长度
    batch_size: int = 8    # 输入批次

    n_layer: int = 8      # 模型层数 8
    n_head: int = 8       # num_heads
    n_embd: int = 768     # hidden_size
    hidden_dim: int = n_embd
    dropout: float = 0.13

    # headers
    head_size: int = n_embd // n_head
    vocab_size: int = len(Encoder)

    # scan
    residual_scale: float = 0.7


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)

        # 注册 attention_mask
        self.register_buffer('attention_mask', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        # q、k、v - 线性层
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weight = q @ k.transpose(-2, -1)
        # 缩放注意力权重
        weight = weight / math.sqrt(self.config.head_size)
        # 注册填充 attention_mask -inf
        weight = weight.masked_fill(self.attention_mask[:seq_len, :seq_len] == 0, float('-inf'))
        # 概率化注意力权重
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        return weight @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)

        return output


# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),  # 升维
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, 3 * config.hidden_dim),  # 中间层
            nn.GELU(),
            nn.Linear(3 * config.hidden_dim, 1 * config.hidden_dim),  # 中间层
            # nn.GELU(),
            # nn.Linear(3 * config.hidden_dim, config.hidden_dim),  # 中间层
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffd = FeedForward(config)

        # 层归一化
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

        # 残差连接 - 缩放因子 - 暂时去掉了
        scale_init = config.residual_scale / math.sqrt(config.n_layer)
        self.att_scale = nn.Parameter(torch.ones(1) * scale_init)
        self.ffd_scale = nn.Parameter(torch.ones(1) * scale_init)

    def forward(self, x):
        # 残差连接 = 原始输入 + 多头注意力结果
        x = x + self.att_scale * self.att(self.ln1(x))  # 1. 保留原始特征
        # 残差连接 = 神经网络之前 + 神经网络之后
        x = x + self.ffd_scale * self.ffd(self.ln2(x))  # 2. 累计特征信息
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embedding, position, norm, mlp, block
        self.config = config
        self.enc = Encoder
        self.block_size = config.block_size

        # 整合词嵌入（带权重绑定）
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置编码
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Transformer Block（保留Pre-LN架构）
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])

        # 输入层归一化
        self.ln_final = nn.LayerNorm(config.n_embd)
        # 输出层（带权重绑定）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)

    # 位置编码 sin cos, 不参与训练
    def create_sin_cos_embedding(self):
        position = torch.arange(self.block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.config.n_embd, 2) * (-math.log(10000.0) / self.config.n_embd))
        pos = torch.zeros(self.config.block_size, self.config.n_embd)
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        return pos

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 初始化为正态分布
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx 输入 tokenIds，targets 目标 tokenIds
        batch, seq_len = idx.size()
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(
            # 确保 位置编码 与 输入 tokenIds 长度一致
            torch.arange(seq_len, device=idx.device)
        )

        # 原始token + pos位置编码
        x = token_emb + pos_emb / math.sqrt(self.config.n_embd)
        # Transformer Block
        x = self.blocks(x)

        # 最终层归一化
        x = self.ln_final(x)
        # 输出层 -> 映射到词向量空间
        logits = self.lm_head(x)

        # 交叉熵损失
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # 我吃 反 shit
    # [ 0.55, 0.45 ] / temperature

    def generate(self, text, max_new_tokens, device, temperature=1.0):
        print("-------------")
        print(text, end='', flush=True)
        # 编码输入文本

        idx = torch.tensor([self.enc.encode(text)], dtype=torch.long, device=device)
        max_size = self.block_size - 1
        for _ in range(max_new_tokens):
            # 防止序列太长，只取最后 block_size 个token
            idx_cond = idx if idx.size(1) <= max_size else idx[:, -max_size:]
            # print(idx_cond)

            # 向前计算，获取预测
            logits, _ = self(idx_cond)
            # 取出最后一步
            logits = logits[:, -1, :]

            # 调整温度
            logits = logits - logits.max(dim=-1, keepdim=True).values
            logits = logits / temperature

            # 生成概率分布
            probs = F.softmax(logits, dim=-1)

            # 采样下一个结果，贪婪算法
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接新 token
            idx = torch.cat((idx, idx_next), dim=1)

            txt = self.enc.decode([idx_next[0].cpu().item()])

            if txt == '。' or txt == '？' or txt == '！' or txt == '”' or txt == '’' or txt == '[UNK]':
                txt += '\n'
            # 打印预测结果
            print(txt, end='', flush=True)

        print("\n", "-------------")
        return idx


class MyDataset(Dataset):
    def __init__(self, path, block_size):
        # 初始化分词器
        self.enc = Encoder
        self.block_size = block_size

        # 自定义终止符
        self.eos_token = self.enc.eos_token_id

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            # 编码文本 text -> tokenIds
            full_encoded = self.enc.encode(text)

            # 训练编码 text -> tokenIds -> [ tokenId, tokenId,... ] (且符合 batch_size)
            self.encoded_data = []
            # 切割文本长度，保留一位做 eos
            splice_len = self.block_size + 1
            for i in range(0, len(full_encoded), self.block_size):
                # 切割 block_size 个文本
                chunk = full_encoded[i:i + self.block_size]
                # 实际序列，需要 +1，以 eos占位，用以满足 x，y错位，从而获得 loss
                if len(chunk) < splice_len:
                    chunk = chunk + [self.eos_token] * \
                        (splice_len - len(chunk))

                self.encoded_data.append(chunk)

            print(f"Total Batch: {block_size}", len(self.encoded_data))

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        # input X
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # target Y
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        # 编码 文本 -> tokenIds
        return self.enc.encode(text)

    def decode(self, tokenIds):
        # 解码 tokenIds -> 文本
        return self.enc.decode(tokenIds)


def maxBatchSize(model, config, device):
    model.eval()
    safe_bs = 0

    for bs in range(1, 10):
        try:
            dummy_input = torch.zeros(
                (bs, config.block_size)).long().to(device)
            with torch.no_grad():
                logits = model(dummy_input)

            # 更新 batch_size
            safe_bs = bs

            # 释放显存
            del dummy_input, logits
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                break

    print(f"Safe Batch Size: {safe_bs}")
