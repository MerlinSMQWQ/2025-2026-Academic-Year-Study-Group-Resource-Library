# 2025-2026-Academic-Year-Study-Group-Resource-Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

2025-2026学年前沿学习小组资料库

## 目录

- [项目介绍](#项目介绍)
- [文件结构](#文件结构)
- [内容概览](#内容概览)
- [如何使用](#如何使用)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [致谢](#致谢)

## 项目介绍

这个资料库用于存放小组内分享的资料，目前设置为public，即使是非小组内的成员也可以浏览或者贡献，欢迎你的贡献，让我们的资料越来越丰富！！！

本项目是一个学术资源分享平台，旨在汇集前沿学习资料，促进知识共享和技术交流。我们专注于人工智能、机器学习、自然语言处理、强化学习等领域的学习资源整理和分享。

## 文件结构

```
2025-2026-Academic-Year-Study-Group-Resource-Library/
├── ComputerScience
│   └── 计算机科学相关书籍.md
├── GitHub操作指南
│   └── GitHub PR操作简明指南.pdf
├── LLM与NLP
│   ├── LLM与NLP相关书籍.md
│   └── Transformer
│       ├── README.md
│       ├── Transformer_demo
│       │   └── translation.ipynb
│       └── 论文与PPT
│           ├── Annotated Transformer.pdf
│           ├── NIPS-2017-attention-is-all-you-need-Paper.pdf
│           └── Transformer讲解.pptx
├── RAG
│   ├── rag_demo_20251026
│   │   ├── RAG知识讲解_20251026.pdf
│   │   ├── README.md
│   │   ├── data
│   │   │   └── GitHubPR操作简明指南.pdf
│   │   ├── log
│   │   │   └── rag_demo_20251026_124139.log
│   │   ├── rag_demo.py
│   │   ├── requirements.txt
│   │   └── vector_database
│   │       ├── index.faiss
│   │       └── index.pkl
│   └── 论文
│       └── RAG.md
├── README.md
└── RL
    ├── books
    │   └── RL_books.md
    └── value-based
        ├── README.md
        ├── env_quick_create.py
        ├── pyproject.toml
        ├── srcCliffWalking_QLearning.py
        ├── srcCliffWalking_sarsa.py
        └── uv.lock
```

## 内容概览

### GitHub操作指南
- GitHub PR操作简明指南.pdf: 介绍如何进行GitHub Pull Request操作

### LLM与NLP
- LLM与NLP相关书籍.md: 大语言模型与自然语言处理相关书籍推荐

### RAG
- RAG论文资料整理
- RAG演示代码及文档
- RAG知识讲解资料

### RL (强化学习)
- 强化学习相关书籍推荐
- 基于价值的强化学习算法实现 (Q-Learning, SARSA)

### Transformer
- Transformer架构相关论文与PPT
- Transformer模型演示代码

## 如何使用

1. 克隆此仓库到本地：
   ```bash
   git clone https://github.com/MerlinSMQWQ/2025-2026-Academic-Year-Study-Group-Resource-Library.git
   ```

2. 浏览相应目录下的学习资料

3. 运行代码示例（如RAG演示）：
   ```bash
   cd RAG/rag_demo_20251026
   pip install -r requirements.txt
   python rag_demo.py
   ```

## 贡献指南

我们欢迎任何形式的贡献！请遵循以下步骤：

### 提交规范

关于commit message的一些写法说明：
commit message要尽量做到清晰，一个清晰的commit message应该包括三个部分：操作类型、作用域、具体信息。
举个例子，当我给项目新添加了一个功能，我在commit message上应该写上

```bash
feat(XXX): added a new feature to XXX
```

#### 操作类型
这里有一个列表，列出了几乎所有操作类型：

| 操作类型 | 说明 |  
| -------- | -------- |
| feat | 新增功能 |  
| fix | 修复bug |
| refactor | 重构，既不增加新的功能，也不修改任何bug |
| docs | 增加或修改文档，比如README.md |
| style | 修改代码风格，但不影响功能 |
| test | 添加测试用例，或者修改测试 |
| chore | 杂项，比如修改或添加.gitignore文件、cmake文件等构件脚本 |
| perf | 性能优化 |
| ci | CI/CD相关的改动 |
| build | 构建方式或者依赖的改动 |
| revert | 回滚某个提交 |

#### 作用域
一般说清楚是哪个模块或者哪一层即可，比如修改了一个项目的路由，作用域可以写为route

#### 具体信息
尽量详细地描述具体的改动，方便后续的追踪和理解

### 贡献流程

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'feat(scope): Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) - 请查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

感谢所有为本项目贡献资源和知识的成员们，你们的分享让这个资料库变得更加丰富和有价值。

特别感谢项目维护者对资料库的持续更新和维护工作。

如果你觉得这个项目对你有帮助，请考虑 Star 一下！ ⭐