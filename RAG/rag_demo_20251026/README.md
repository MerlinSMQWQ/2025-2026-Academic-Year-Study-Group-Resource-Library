# 💻 rag_demo_20251026 项目架构

rag_demo/
│
├── data/
│   └── GitHubPR操作简明指南.pdf      # 知识文件
│
├── models/
│   └── bge-large-zh-v1___5          # Embedding模型
│   └── bge-reranker-base            # Rerank模型
│
├── vector_database/                 # 向量数据库
│   ├── index.faiss                  # 向量索引文件（二进制）
│   └── index.pkl                    # 元数据文件
│
├── log/                             # 存放日志
│
├── rag_demo.py                      # 主程序
├── requirements.txt                 # 环境依赖
├── README.md                        # 说明文档
└── RAG知识讲解_20251026.pdf          # 组会讲解文档



# 🔍 模型下载方式与命令

⚠️ 在models文件夹下，只有两个空的模型文件，由于要上传到github仓库，文件太大会导致上传失败，所以大家需要自己下载后才可运行ragg_demo.py。

本项目中的 Embedding 和 Rerank 模型使用“modelscope魔搭社区”下载。
ps：魔搭社区就相当于国产版的huggingface，不需要使用梯子，下载速度比huggingface快很多。

详细的下载方式与指令官网有讲解：
https://www.modelscope.cn/docs/models/download

下载 Embedding 模型命令：
modelscope download --model 'BAAI/bge-large-zh-v1.5' --local_dir '/root/autodl-tmp/QH/rag_demo_20251026/models/bge-large-zh-v1___5'

下载 Rerank 模型命令：
modelscope download --model 'BAAI/bge-reranker-base' --local_dir '/root/autodl-tmp/QH/rag_demo_20251026/models/bge-reranker-base'



# 💡 简要解释说明：

1. 向量数据库说明

Faiss 是 Facebook 开源的一个专门用于高效相似性搜索和聚类的库，能够快速处理大规模数据，并支持在高维空间中进行相似性搜索。Faiss的核心功能是将候选向量集封装成一个 index 数据库，加速检索相似向量 Top K 的过程。部分算法是在 GPU 上实现的，以充分利用 GPU 的并行计算能力。


文件名	           内容类型	                              作用	                                         说明
index.faiss	  向量索引文件（二进制）	      保存每个文档 chunk 的向量表示 + 索引结构	   用于高效相似度搜索（FAISS 内部格式）（不可读）
index.pkl	  元数据文件（Python pickle）    保存每个向量对应的原始文档信息	              用于返回真实文本、页码、来源等（可读）

感兴趣的同学可以上网搜集资料学习一下faiss，当然创建向量数据库还有很多别的方法，比如：chroma、milvus等。


2. 项目环境搭建

推荐大家使用 uv 来创建管理自己的项目。
官网：https://uv.doczh.com/


3. LLM的调用

在这个demo中，我是通过硅基流动这个第三方来调用LLM的，需要自己充费生成密钥（新用户注册时会送10块钱，先把这些钱用了再充值😆）。
官网：https://www.siliconflow.cn/
当然，使用LLM还有别的方法，可根据情况自行选择。


4. 租用GPU

我是在 AutoDL 这个算力平台租用的GPU，对比国内其他家平台，这个平台目前来说价格很合适，也很好维护，推荐大家使用。
AutoDL 是一款专业的 GPU 租用平台，提供了从入门级到高端的 GPU 配置选择，涵盖了深度学习所需的多种主流显卡型号，如 RTX 3090、RTX 4090、NVIDIA A100 等，广泛应用于深度学习、计算机视觉、自然语言处理等领域。平台支持按需租用、按时计费，用户可以灵活选择适合的配置，节省成本。
官网：https://www.autodl.com/
大家可以上网搜索资料进行使用。



# ⭐ 请注意：

1. 本项目只是一个最简单版本的基于RAG技术的系统，在真实项目中，每一个步骤都需要更仔细的处理，比如数据如何处理、并发量为多少、把 “检索+重排” 这部分单独抽成多个独立子进程池、前端怎么设计等。
2. 如果大家在看的过程中发现本项目文档及代码有任何问题，请及时反馈，我立刻修改。
3. 感谢大家的阅读~😉