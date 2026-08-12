## 项目结构

```
learn-llm-by-building/
├── tokenizer/                # 分词器相关目录
│   ├── tokenizer.json        # 分词词表数据
│   └── tokenizer_config.json # 分词器配置参数
├── llm.py                    # 模型主体定义：实现Transformer、多头注意力、前馈网络等模块
├── dataloader.py             # 数据加载模块：文本读取、分词、构建训练批次、序列预处理
├── train.py                  # 训练主脚本：执行训练循环、优化器配置、损失计算、模型保存
├── chat.py                   # 推理对话脚本：加载训练好的权重，实现交互式文本生成
├── LICENSE
└── README.md

```
