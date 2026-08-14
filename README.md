<p align="center">
  <img src="./media/LLM.png" width="100%">
</p>

# 项目结构

```
learn-llm-by-building/
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── llm.py
├── dataloader.py
├── train.py
├── chat.py
├── LICENSE
├── README.md
└── media
```

# 模块功能说明

1. **tokenizer（分词器目录）**
    负责文本编码与解码：将自然语言文本转换为模型可识别的token序列；同时支持推理阶段，把模型输出的token还原为可读文本。包含词表文件与分词器配置。

2. **llm.py**
    模型主体文件。基于Transformer架构搭建基础大模型，定义模型前向传播逻辑。

3. **dataloader.py**
    数据集加载与预处理模块。读取原始文本，调用分词器完成文本token化，构建训练样本、划分序列长度、生成批次数据，为训练提供源源不断的输入。

4. **train.py**
    训练脚本。初始化模型、数据集、优化器与损失函数，搭建完整训练循环；执行前向传播、梯度更新。

5. **chat.py**
    交互式推理脚本。加载训练完成的模型权重；提供对话交互入口。
***
执行流程：原始文本 → tokenizer编码 → dataloader封装批次 → train.py训练模型 → chat.py加载权重进行对话推理

# llm.py 模型架构
## 1.PositionalEncoding
$$
\begin{align*}
PE_{(pos, 2i)} &= \sin\left(pos \cdot 10000^{- \frac{2i}{d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(pos \cdot 10000^{- \frac{2i}{d_{\text{model}}}}\right)
\end{align*}
$$
### 符号定义
1. pos

当前 token 在句子里的绝对位置，从 0,1,2,3... 依次递增。

例：一句话 ```我 爱 编程```，```我``` 的 pos=0，```爱``` pos=1，```编程``` pos=2。

2. dmodel

词嵌入向量总维度，本项目设置 dmodel = 512。

每个 token 会被映射成长度为 dmodel 的浮点向量，位置编码和嵌入向量维度一致，才能相加融合。

3. i

向量内成对维度组的索引，取值 0,1,2,3...

### 位置编码公式的数学解释

$$
\begin{aligned}
\sin(a + b) &= \sin a \cos b + \cos a \sin b \\
\cos(a + b) &= \cos a \cos b - \sin a \sin b
\end{aligned}
$$

假设两个位置差值为 k，即位置 pos + k 相对 pos 偏移 k：

$$
\begin{aligned}
PE(pos + k, 2i) &= \sin \big((pos + k) \cdot \omega_i\big) = \sin(pos \cdot \omega_i + k \cdot \omega_i) \\
PE(pos + k, 2i+1) &= \cos \big((pos + k) \cdot \omega_i\big) = \cos(pos \cdot \omega_i + k \cdot \omega_i)
\end{aligned}
$$

$$
\omega_i = 10000^{-\frac{2i}{d_{\text{model}}}}
$$

展开后能看到：位置 pos + k 的编码向量，可以由 pos 的编码向量、偏移量 k 的编码向量线性组合算出。
这意味着模型只需要简单线性变换，就能识别任意两个 token 的相对距离，不需要学习位置参数。

频率 $\omega_i$ 随 $i$ 增大单调递减：
- 小 $i$（前半段维度）： $\omega_i$ 大 → 三角函数周期极长
  周期长 = 数值变化缓慢，用来捕捉**远距离依赖**（句子开头和结尾的关联）
- 大 $i$（后半段维度）： $\omega_i$ 小 → 三角函数周期很短
  周期短 = 数值快速波动，用来捕捉**局部相邻词语**的细微位置差异

把不同频率正弦/余弦塞进向量不同维度，相当于给位置信号做**多尺度频谱分解**，长短距离信息全部存进同一个向量。

### 代码实现
```
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
```
### 代码解析
```
self.dropout = nn.Dropout(dropout)
```
训练阶段随机置零输入张量部分元素并缩放剩余数值，抑制过拟合；`model.eval()`推理模式下自动关闭丢弃逻辑。参数`dropout`代表元素置零概率，本项目设置 0.1。

```
pe = torch.zeros(max_seq_len, d_model)
```
创建形状`[max_seq_len, d_model]`全零张量，预存储所有位置对应的编码向量。

```
pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
```
- `torch.arange(0, max_seq_len)`：生成 0 到最大序列长度 - 1 的位置整数
- `.unsqueeze(1)`：一维向量转为列向量`[T,1]`，适配广播乘法
- `.float()`：转换浮点类型，三角函数仅支持浮点张量运算

```
div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div)
```
数学公式实现

`[:, 0::2]`

竖直方向（行）**全部保留不截断**

水平方向（列）**做间隔采样，只取偶数列**

```
self.register_buffer('pe', pe.unsqueeze(0))
```
PyTorch `register_buffer(name, tensor)`接口解析：

1. buffer 张量不属于可训练参数`nn.Parameter`，不参与反向传播梯度更新；

2. 模型保存 / 加载时 buffer 会随权重文件一同存储；

3. `pe.unsqueeze(0)`新增 batch 维度，形状从`[T,C]`变为`[1,T,C]`，可直接和任意 batch 输入广播相加。

### 前向传播代码解析
```
def forward(self, x):
    return self.dropout(x + self.pe[:, :x.size(1)])
```
输入`x`：词嵌入张量，shape = `[batch_size, seq_len, d_model]`

1. `self.pe[:, :x.size(1)]`
截取预计算位置编码前`seq_len`长度，适配当前输入真实序列长度，防止索引越界；

2. `x + self.pe[:, :x.size(1)]`
广播加法，将时序位置信息融合进词嵌入向量；

3. `self.dropout(...)`
叠加 dropout 正则，降低位置编码带来的过拟合风险；

4. 返回融合位置信息后的嵌入张量，传入后续多头注意力模块。

### 补充内容
pe 原本形状：[T, C]

pe.unsqueeze(0) → 新增第 0 维，变成 三维张量 [1, T, C]

T：序列长度（token 数量）
C：dmodel 嵌入维度

什么是广播（broadcasting）？

> 
> 广播 : PyTorch/Numpy 的自动机制：**两个形状不完全一致的张量做加减乘除时，自动复制扩展维度，让两者形状匹配，再运算**
**广播相加 : 依靠广播机制执行加法**
```
a = torch.tensor([1,2,3])
b = torch.tensor([10])
print(a + b)
```
```
[11,12,13]
```
逻辑：自动把 `b` 复制成 `[10,10,10]` 再相加

**广播相乘 : 依靠广播机制执行乘法**
```
pos = torch.arange(5).float()       # shape [5]
pos_col = pos.unsqueeze(1)          # shape [5, 1]
div = torch.arange(4).float()       # shape [4]

res = pos_col * div
print(res.shape)    # 输出 torch.Size([5, 4])
```
```
pos_col = [[0.],
           [1.],
           [2.],
           [3.],
           [4.]]
div = [0., 1., 2., 3.]

相乘结果：
[[0., 0., 0., 0.],
 [0., 1., 2., 3.],
 [0., 2., 4., 6.],
 [0., 3., 6., 9.],
 [0., 4., 8., 12.]]
```
## 2.MultiHeadAttention
$$
\begin{aligned}
\text{Attention}(Q,K,V) &= \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \\
\text{head}_i &= \text{Attention}\big(QW_i^Q,\;KW_i^K,\;VW_i^V\big) \\
\text{MultiHead}(Q,K,V) &= \text{Concat}\big(\text{head}_1,\text{head}_2,\dots,\text{head}_h\big)W^O
\end{aligned}
$$

### 符号定义
1. Q,K,V — Query、Key、Value

- Q（Query 查询）：代表当前 token 想要寻找什么信息
- K（Key 键）：代表所有 token 具备什么信息
- V（Value 值）：代表所有 token 携带的内容信息

在自注意力（Self-Attention）中，Q,K,V 来自同一输入 x。

2. $W_i^Q,W_i^K,W_i^V$ — 分头投影权重矩阵

- 输入向量 $x$：维度 $d_{\text{model}}$
- 投影矩阵 $W_i^Q$：行数=输入维度，列数=输出维度
- 输出 $q_i$：维度 $d_k$
矩阵乘法完成一次线性空间变换，这个变换就叫**线性投影**。


3. $QK^\top$ 矩阵乘法得到相似度分数矩阵，矩阵中元素代表第 t 个 Token 和第 s 个 Token 的关联程度。

4. $\text{softmax}(z_j) = \frac{\exp(z_j)}{\sum_i \exp(z_i)}$ 对相似度矩阵最后一维归一化，将原始分数转换为总和为 1 的注意力权重分布，权重越大代表该 Token 越重要。

5. Concat

   把 h 个独立注意力头输出，在特征维度拼接。
   单个头输出形状 [B,T,d_k]，拼接后恢复为 [B,T, $d_{\text{model}}$ ]，还原原始维度。
   并非向量相加，而是向量首尾横向拼接，过程中维度不变

### 代码实现
```
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)
```
### 代码解析
```
self.q_proj = nn.Linear(d_model, d_model)
self.k_proj = nn.Linear(d_model, d_model)
self.v_proj = nn.Linear(d_model, d_model)
```
Pytorch nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)

- `in_features`：输入特征的**一维长度**
- `out_features`：输出特征的**一维长度**

用法：

1.创建了一个 nn.Linear 模块实例；

2.在模块内部自动分配两块可学习参数张量：

weight：形状 [out_features, in_features] = [d_model, d_model]，也就是 $W_v$

bias：形状 [d_model]，也就是偏置 $b_v$

3.用默认策略随机填充 weight 和 bias；把这个模块赋值给 self.v_proj

```
self.o_proj = nn.Linear(d_model, d_model)
```
o_proj 的作用：把多个头的信息融合、重组、加权整合。
### 前向传播代码解析
```
def forward(self, x):
        B, T, C = x.shape
```

输入 x: [B, T, C]

B = batch_size 批次大小
    
T = sequence length 序列长度（token数量）
    
C = d_model 模型隐层维度
```
q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
```
```
self.q_proj(x)
```
执行公式

$$Q_{all} = X W_q^\top + b_q$$

实际进行的矩阵运算：

$$
\begin{bmatrix}y_1 \\
y_2\end{bmatrix} = \begin{bmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23}
\end{bmatrix}\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix} + \begin{bmatrix}b_1 \\
b_2\end{bmatrix}
$$

输出形状：`[B, T, d_model]`
```
.view(B, T, self.n_heads, self.d_k)
```
等价 reshape：`d_model = n_heads * d_k`

形状变为：`[B, T, n_heads, d_k]`
```
.transpose(1, 2)
```
交换维度 1 和 2，矩阵变为：`[B, n_heads, T, d_k]`

为什么交换？

方便后续批量矩阵乘法：**每个头独立并行做注意力计算**

维度含义：`[batch, head_num, token_num, per_head_dim]`

此时 q,k,v 形状统一：`[B, n_heads, T, d_k]`
```
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```
- `k.transpose(-2, -1)`：`[B, h, T, d_k] → [B, h, d_k, T]`
- `torch.matmul(q, k.transpose(-2,-1))`
矩阵乘法：`[B,h,T,d_k] @ [B,h,d_k,T] = [B,h,T,T]`
```
mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```
`.triu(torch.ones(T, T, device=x.device), diagonal=1)`：上三角矩阵，**对角线右上区域 = True**
矩阵位置 $(i,j)$ ：

$j$ > $i$ ：mask=True → 未来 token 不允许被看见
把对应位置分数填充 -inf，经过 softmax 后权重趋近 0，实现不能偷看未来
```
attn = self.dropout(F.softmax(scores, dim=-1))
```
- `softmax(dim=-1)`：对**每一行（查询 token）**，所有 key token 权重归一化，总和 = 1；
- `self.dropout`：注意力权重随机置零，正则化，防止过拟合。
`attn` 形状：`[B, n_heads, T, T]`，就是注意力权重矩阵。
```
out = torch.matmul(attn, v)
```
执行了运算：

[B,h,T,T] @ [B,h,T,d_k] = [B,h,T,d_k]

`torch.matmul(a, b)`:

**张量矩阵乘法**，根据两个张量维度自动选择不同运算规则。

数学上： $C = AB$
```
out = out.transpose(1, 2).contiguous().view(B, T, C)
```
`.transpose(1,2)`：`[B,h,T,d_k] → [B,T,h,d_k]`

`.view(B, T, C)`

把 `[B,T,h,d_k]` 压扁回 `[B,T,h*d_k] = [B,T,C]`

### 过程总结

输入 X 分别投影得到 $Q_{all},K_{all},V_{all}$

Q、K 计算相似度分数 $\dfrac{QK^\top}{\sqrt{d_k}}$

softmax 得到注意力权重 $\text{attn}$

注意力权重和 V 做矩阵乘法： $\text{head}_i = \text{softmax}(...)V_i$

这里只有 attn × V，全程没有 Q+V、K+V

所有 head 结果 Concat 拼接 得到 $Z_{\text{concat}}$

拼接结果送入输出投影层：

$$\boldsymbol{O} = Z_{\text{concat}} W_o^\top + b_o$$

## 3.FeedForward
$$
\text{FFN}(X) = \boldsymbol{W_2}\cdot \text{Dropout}\Big(\text{ReLU}\big(X\boldsymbol{W_1}^\top + \boldsymbol{b_1}\big)\Big) + \boldsymbol{b_2}
$$
### 代码实现
```
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
```
### 代码解析
```
self.fc1 = nn.Linear(d_model, d_ff)
```
升维：

把隐层维度由 $d_{\text{model}}$ 变为 $d_{\text{ff}}$

一般 $d_{\text{ff}} = 4 * d_{\text{model}}$
```
self.dropout = nn.Dropout(dropout)
```
降维
### 前向传播代码解析
```
def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
```
执行公式

$$
\text{FFN}(X) = \boldsymbol{W_2}\cdot \text{Dropout}\Big(\text{ReLU}\big(X\boldsymbol{W_1}^\top + \boldsymbol{b_1}\big)\Big) + \boldsymbol{b_2}
$$
