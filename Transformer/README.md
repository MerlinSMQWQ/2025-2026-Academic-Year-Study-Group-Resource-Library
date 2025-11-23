# Transformer 学习

这个仓库包含两部分内容：一是 Transformer 相关的两篇经典论文与讲解 PPT，二是一个展示 Transformer 在机器翻译（中译英）中应用的极简 Demo。

## 文件夹说明

### 1. `论文与PPT`
- **论文1**: [Attention Is All You Need](论文与PPT/NIPS-2017-attention-is-all-you-need-Paper.pdf) (Transformer 原始论文)
- **论文2**: [Annotated Transformer.pdf](论文与PPT/Annotated Transformer.pdf) (带注释的Transformer )
- **PPT**: [Transformer 技术讲解](论文与PPT/Transformer讲解.pptx) (讲解PPT)

### 2. `Transformer_demo`
- 一个极简的 Transformer 机器翻译 Demo，展示如何用 Transformer 实现翻译任务。


## Transformer_demo 说明

这个 Demo 聚焦于**展示 Transformer 在机器翻译任务中的核心流程**，并非追求实际翻译效果。

### 功能
- 实现了完整的 Transformer 架构（编码器+解码器）。
- 包含数据预处理、模型训练、贪心解码推理的全流程。
- 支持将中文句子翻译成英文句子。

### 技术细节
- **依赖**：PyTorch >= 2.0，无需额外安装其他库。
- **数据集**：使用极简的玩具数据集（为了快速演示，数据规模很小）。
- **模型设计**：包含词嵌入、位置编码、多头注意力、前馈网络等 Transformer 核心模块。
- **训练与推理**：实现了完整的训练循环和贪心解码推理函数。

### 局限性
由于 Demo 采用了**极简数据集**，训练后模型的损失率不会很低，实际翻译测试结果也不够理想。这个 Demo 的核心价值是**展示 Transformer 实现机器翻译的流程与代码逻辑**，而非追求工业级翻译效果。

### 快速启动
1. 确保安装 PyTorch >= 2.0。
2. 进入 `Transformer_demo` 文件夹，运行主程序即可启动训练与测试。


## 学习建议
- 先通过 `论文与PPT` 文件夹学习 Transformer 原理，这里推荐一篇博客[Transformer 模型详解](https://blog.csdn.net/benzhujie1245com/article/details/117173090)。
- 阅读 `Transformer_demo` 中的代码，理解如何将理论落地为机器翻译任务。
- 若需提升翻译效果，可尝试替换为真实的大规模平行语料、调优模型参数或改用更复杂的解码策略。