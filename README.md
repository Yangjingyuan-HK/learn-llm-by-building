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
