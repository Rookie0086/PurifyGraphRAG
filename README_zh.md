# PurifyGraphRAG

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://www.docker.com/"><img alt="Docker" src="https://img.shields.io/badge/Docker-Required-2496ED?style=flat-square&logo=docker&logoColor=white"></a>
  <a href="https://milvus.io/"><img alt="Milvus" src="https://img.shields.io/badge/Milvus-Vector_DB-00D1B2?style=flat-square"></a>
  <a href="https://www.nebula-graph.io/"><img alt="NebulaGraph" src="https://img.shields.io/badge/NebulaGraph-Graph_DB-4169E1?style=flat-square"></a>
  <img alt="Status: Under Review" src="https://img.shields.io/badge/Status-Under_Review-cf4f4f?style=flat-square">
</p>

> [English](./README.md) · **中文**

PurifyGraphRAG 是一个在**增量知识摄入过程中完成知识净化**的图检索增强生成（Graph-based RAG）框架。框架基于**自净化双层图架构**实现高效的图索引与检索：L1 为有界的瞬态内存图（NetworkX，LRU+TTL 与引用计数淘汰），负责暂存高频访问的知识；L2 为持久化知识图（NebulaGraph），负责固化高价值子图；底层由全量向量库（Milvus）兜底。访问频率驱动的晋升机制（知识净化，Knowledge Purification）在晋升阶段过滤抽取噪声、抑制图的无限膨胀。

---

## 📖 背景与动机

将知识图谱与检索增强生成（RAG）结合，可以显著增强大语言模型（LLM）的复杂推理能力。然而，在以**高频流式文档摄入**与**复杂多跳问答**为特征的真实部署环境中，现有 Graph-based RAG 系统暴露出三个根本性瓶颈：

1. **图索引与实体对齐开销过高**：主流方法在三元组抽取和全局实体消歧阶段依赖同步的 LLM 调用，接近 O(N²) 的调用复杂度在持续数据流下引发严重的更新延迟与成本问题。
2. **图拓扑污染与无界膨胀**：传统系统普遍采用"急写（eager write）"策略，将抽取到的全部实体和关系直接写入持久图数据库。随着流式文档不断摄入，图迅速稠密化，检索时在高度数超节点周围引发邻域爆炸，注入大量长尾噪声。
3. **被动一次式检索的语义漂移**：静态检索流程无法根据已积累的中间线索自适应调整搜索意图，遇到拓扑中的关系缺口时容易漂移到无关分支，导致多跳推理链断裂。

针对上述瓶颈，PurifyGraphRAG 引入三个相互协同的机制：

1. **异步流水线索引引擎**：以两个并发阶段解耦高延迟 LLM 符号抽取与向量生成，配合 **LLM-free 的漏斗式实体对齐**（级联相似度阈值 + 别名约束），以极低开销完成图片段拼接，实现低延迟图构建。
2. **自净化双层图架构**：L1 瞬态内存图受硬容量约束（LRU+TTL + 引用计数垃圾回收），L2 仅固化通过访问频次阈值 `h(c) >= tau_hit` 的高价值子图（知识净化），底层由全量向量库兜底；**懒重水化**（Lazy Rehydration）在 L1 未命中时无需 LLM 调用即可毫秒级恢复长尾连通性。
3. **迭代式 Agentic 查询分解器**：将 LLM 实例化为主动规划智能体，在图拓扑上执行带跳衰减约束（γ^l）与确定性循环打断的 Beam Search，依据中间状态动态规划下一跳探索路径，抑制多分支拓扑发散带来的指数级噪声传播。

**主要实验结论**：在 RGB、2WikiMultihopQA、HotpotQA 及自建的 SpecificQA 增量更新基准上，PurifyGraphRAG 取得了领先的端到端 QA 准确率；在支持增量更新的基线中取得最低的索引与更新延迟（各数据集索引时间较最快基线降低 **3.4%–35.4%**），并将持久化拓扑占用（节点/边数量）相比最节省空间的基线系统压缩 **81.1%–94.7%**。

---

## 🏗️ 核心架构

![PurifyGraphRAG 核心架构](framework.jpg)

## 📁 项目结构

```
PurifyGraphRAG/
├── src/
│   ├── PurifyGraphRAG.py          # 主入口 + CLI
│   ├── pipeline.py                # 文档入库管线 (asyncio.gather；合并嵌入请求 + 批式离线对齐)
│   ├── memory_graph.py            # L1 瞬态内存图 (L1CachePolicy: LRU+TTL) + L2 NebulaGraph
│   │                                 # rehydrate_chunk_from_milvus() 实现懒重水化 (Lazy Rehydration)
│   ├── retriever.py               # HybridRetriever (configurable gamma/B/max_hops)
│   ├── IterativeAgenticEngine.py  # IterativeAgenticEngine (code-level loop-breaking)
│   ├── retrieval/
│   │   ├── fusion.py              # RRF / Weighted / Dual fusion (k parameter)
│   │   └── reranker.py            # API / Local reranker
│   ├── entity/
│   │   ├── resolver.py            # AsyncEntityResolver (Funnel Alignment: tau_sim, tau_desc)
│   │   └── bert_sim.py            # BERTScore (all-MiniLM-L6-v2)
│   ├── llm/
│   │   └── env.py                 # LLM + Embedding environment (OpenAI/DeepSeek/Ollama)
│   ├── eval.py                    # Evaluation metrics (EM, ROUGE-L, BERTScore)
│   └── utils/
│       ├── prompts.py             # Prompt templates
│       ├── logger.py              # Pipeline logger
│       ├── llm_cache.py           # LLM response cache
│       └── base.py                # Base utilities (config, JSON, checkanswer)
├── database/
│   ├── milvus.py                  # Milvus vector DB wrapper
│   ├── nebulagraph.py             # NebulaGraph client wrapper
│   └── setup/                     # Docker install scripts
├── data/                          # Dataset loaders + datasets
├── config/
│   └── config.yaml                # Global configuration
├── scripts/
│   ├── storage_analysis.py        # 存储占用分析（Table 5 / 存储瓶颈验证）
│   ├── efficiency_comparison.py   # 效率对比（延迟 + Token 消耗 + LLM 调用次数）
│   └── fair_evaluation.py         # Fig. 5 公平评估复现（从空 L1+L2 出发）
├── requirements.txt
└── README.md
```

## ⚙️ 环境与部署

### 依赖

- Python >= 3.10
- Docker（Milvus + NebulaGraph）

### 安装

```bash
git clone https://github.com/Rookie0086/PurifyGraphRAG.git
cd PurifyGraphRAG

conda create -n purifygraphrag python=3.10 -y
conda activate purifygraphrag

pip install -r requirements.txt
```

### 数据库配置

本项目使用 [Milvus](https://github.com/milvus-io/milvus) 作为向量数据库（存储 chunk / entity 向量，作为重水化兜底层），使用 [NebulaGraph](https://github.com/vesoft-inc/nebula) 作为图数据库（L2 持久化存储层）。

两个数据库均以 Docker 方式部署，安装脚本位于 `database/setup/`，无需 sudo（要求当前用户可直接运行 `docker` 与 `docker compose`）。使用以下命令安装并启动：

```bash
# 安装并启动 Milvus（单机版，镜像 milvusdb/milvus:v2.3.9，数据存于 ~/.local/share/milvus）
bash database/setup/milvus-install-user.sh start

# 安装并启动 NebulaGraph（docker-compose 部署，含 NebulaGraph Studio 与 console）
bash database/setup/nebula-install-user.sh install
```

#### 服务管理

```bash
# Milvus：stop 停止 / delete 删除容器与数据
bash database/setup/milvus-install-user.sh stop|delete

# NebulaGraph：start 启动 / stop 停止 / delete 删除（默认工作目录 ~/.nebula-up）
bash database/setup/nebula-install-user.sh start|stop|delete

# NebulaGraph 重启（指定 compose 工作目录）
bash database/setup/nebula-restart-user.sh ~/.nebula-up
```

#### 连接验证

框架启动时会自动检查两个数据库的连通性（失败时会打印上述安装命令并退出），也可手动确认容器状态：

```bash
docker ps | grep -E "milvus|nebula"
```

---

## 🚀 快速开始

### Python API（索引 + 检索 + 回答）

```python
import asyncio
from src.PurifyGraphRAG import PurifyGraphRAG

app = PurifyGraphRAG(
    dataset="hotpotqa",
    l1_max_chunks=200,       # C_max
    promotion_threshold=3,   # tau_hit
)

async def main():
    await app.index(start=0, end=300)
    app.load_dataset(start=0, end=300)
    results = await app.query(
        questions=app.questions[:5],
        start=0, end=5,
        mode="hybrid",
        answer_topk=6,
    )
    for r in results:
        print(f"Q: {r['query']}\nA: {r['predict']}\n")
    app.shutdown()

asyncio.run(main())
```

### 命令行运行

所有配置在 `config/config.yaml` 中设置后运行：

```bash
# 完整流程 (索引 + QA)
python -m src.PurifyGraphRAG

# 仅构建索引 (config.yaml: retrieval.index_only = true)
python -m src.PurifyGraphRAG

# 跳过索引、直接 QA (config.yaml: retrieval.skip_index = true)
python -m src.PurifyGraphRAG
```

---

## 🔬 复现论文实验

### 数据集

论文实验使用四个数据集（三个公开基准 + 一个自建基准）：

| 数据集 | 类型 | 问题数 | 文档数 | Token 数 |
|--------|------|--------|--------|----------|
| RGB | 噪声鲁棒推理 | 300 | 7,976 | 1,649,548 |
| 2WikiMultihopQA | 多跳推理 | 600 | 6,000 | 536,444 |
| HotpotQA | 多跳推理 | 600 | 5,949 | 796,386 |
| SpecificQA | 同名实体消歧 | 600 | 2,066 | 436,552 |

说明：

- **RGB**：包含单跳与多跳查询及噪声文档，用于测试系统抗噪能力。
- **2WikiMultihopQA / HotpotQA**：需要跨文档实体链接与多跳聚合的标准多跳问答基准，遵循先前工作各随机选取 600 条查询。
- **SpecificQA**：基于 WhoQA 构建，通过特征约束的问题重构方法向文档流中人工注入同名异义（描述不同）的对抗实体，专用于压测实体消歧能力与增量更新成本；每条样本包含 `phase_1_data`（干扰实体文档）与 `phase_2_data`（目标实体文档）两阶段流式输入。

### 实验设置

所有超参数集中定义在 `config/config.yaml`，论文符号与配置项映射如下：

| 论文符号 | 配置项 | 默认值 | 说明 |
|---------|--------|-------|------|
| `tau_sim` | `entity_alignment.tau_sim` | 0.85 | 实体向量相似度阈值 |
| `tau_desc` | `entity_alignment.tau_desc` | 0.5 | 实体描述相似度阈值 |
| `tau_hit` | `indexing.tau_hit` | 3 | chunk 晋升阈值 h(c) >= tau_hit |
| `C_max` | `indexing.C_max` | 200 | L1 瞬态层容量上限 (max chunks) |
| `gamma` | `retrieval.gamma` | 0.5 | 跳衰减权重 (gamma^l) |
| `B` | `retrieval.B` | 5 | Beam 宽度 (每跳 top-B 节点) |
| `k` | `fusion.k` | 60 | RRF 融合参数 |

### 端到端 QA 准确率

对每个数据集，修改 `config/config.yaml` 后运行完整流程，再用评估脚本计算 ACC / ROUGE-L / BERTScore：

```yaml
# config/config.yaml —— 以 HotpotQA 为例
data:
  dataset: hotpotqa        # rgb_en_refine | wikimultihopqa | hotpotqa
  start: 0
  end: 600                 # RGB 为 300，2Wiki / HotpotQA 为 600

retrieval:
  nebula_space: hotpotqa   # 与 dataset 保持一致
  chunk_collection: hotpotqa
  entity_index_name: entity_index_hotpotqa
  index_only: false
  skip_index: false
```

```bash
# 1. 运行完整流程（文档索引 + 检索 + 回答）
python -m src.PurifyGraphRAG
# QA 结果保存至 output/qa/qa_results_hotpotqa_0_600.json

# 2. 计算 ACC / ROUGE-L / BERTScore
python -c "from src.eval import evaluate_from_file, print_report; print_report(evaluate_from_file('output/qa/qa_results_hotpotqa_0_600.json', use_rougel=True, use_bert=True))"
```

三个数据集对应的 `dataset` 取值：RGB → `rgb_en_refine`，2Wiki → `wikimultihopqa`，HotpotQA → `hotpotqa`。更换数据集时请同步修改 `nebula_space` / `chunk_collection` / `entity_index_name`。

其余实验都可以在 `scripts/` 中找到实验脚本。

---

## 📊 实验结果

以下为论文报告的部分实验结果（详见论文正文）。

### Table 3：端到端 QA 性能

| 方法 | RGB ACC | RGB R-L | RGB BERT | 2Wiki ACC | 2Wiki R-L | 2Wiki BERT | HotpotQA ACC | HotpotQA R-L | HotpotQA BERT |
|------|---------|---------|----------|-----------|-----------|------------|--------------|--------------|---------------|
| MS-GraphRAG | 94.67 | 0.924 | 0.735 | 36.20 | 0.392 | 0.435 | 42.70 | 0.516 | 0.496 |
| LightRAG | 64.70 | 0.647 | 0.571 | 30.00 | 0.345 | 0.434 | 29.30 | 0.417 | 0.572 |
| HippoRAG | 91.00 | 0.875 | 0.806 | 66.30 | 0.688 | 0.687 | 57.50 | 0.645 | 0.695 |
| HippoRAG2 | **98.70** | 0.732 | 0.846 | 61.80 | 0.645 | 0.702 | 62.70 | 0.705 | 0.764 |
| EraRAG | 73.33 | 0.766 | 0.429 | 49.00 | 0.518 | 0.472 | 36.30 | 0.422 | 0.463 |
| KAG | 97.30 | 0.724 | 0.829 | 70.70 | 0.723 | 0.767 | 68.00 | 0.749 | 0.782 |
| HyperGraphRAG | 97.67 | 0.942 | 0.583 | 39.80 | 0.405 | 0.409 | 51.70 | 0.547 | 0.522 |
| Clue-RAG | 97.67 | 0.940 | 0.853 | 55.50 | 0.508 | 0.643 | 63.17 | 0.611 | 0.729 |
| **PurifyGraphRAG** | 97.67 | **0.948** | **0.860** | **73.17** | **0.753** | **0.780** | **68.30** | **0.760** | **0.800** |

### Table 4：索引时间（秒）

| 方法 | RGB | 2Wiki | HotpotQA | SpecificQA (Index) | SpecificQA (Update) |
|------|------|-------|----------|--------------------|---------------------|
| MS-GraphRAG | 25693.87 | 4811.89 | 4291.40 | 5603.09 | 5487.59 |
| LightRAG | 7171.88 | 2497.70 | 4127.51 | 4052.27 | 2122.83 |
| HippoRAG2 | 6315.58 | 2500.48 | 4690.68 | 1864.41 | 896.83 |
| KAG | 6955.52 | 3077.61 | 3009.99 | 3290.18 | 1073.76 |
| HyperGraphRAG | 20356.20 | 5625.48 | 7046.84 | 3087.18 | 1024.46 |
| **PurifyGraphRAG** | **4560.60** | **2204.72** | **2907.40** | **1363.60** | **579.80** |

### Table 5：持久化拓扑占用（节点 / 边数量）

PurifyGraphRAG\* 为移除双层架构的消融变体。

| 方法 | RGB Node | RGB Edge | 2Wiki Node | 2Wiki Edge | HotpotQA Node | HotpotQA Edge |
|------|----------|----------|------------|------------|---------------|---------------|
| MS-GraphRAG | 22,877 | 34,211 | 12,899 | 11,980 | 8,688 | 7,384 |
| LightRAG | 20,257 | 18,432 | 12,060 | 7,304 | 19,519 | 12,620 |
| HippoRAG | 42,717 | 203,743 | 18,896 | 74,708 | 33,384 | 102,376 |
| HippoRAG2 | 49,095 | 260,440 | 22,203 | 91,717 | 38,868 | 42,928 |
| KAG | 109,820 | 155,554 | 37,007 | 53,093 | 53,369 | 85,684 |
| HyperGraphRAG | 126,360 | 114,151 | 64,787 | 52,167 | 81,581 | 67,240 |
| Clue-RAG | 35,734 | 48,257 | 63,129 | 89,615 | 110,640 | 140,305 |
| PurifyGraphRAG\* | 32,948 | 52,298 | 19,299 | 18,672 | 34,190 | 35,765 |
| **PurifyGraphRAG** | **1,059** | **1,257** | **1,081** | **1,035** | **1,391** | **1,393** |

### SpecificQA：实体消歧性能

| 方法 | ACC | Rouge-L | BERTScore |
|------|------|---------|-----------|
| MS-GraphRAG | 57.83 | 0.413 | 0.568 |
| LightRAG | 25.00 | 0.335 | 0.488 |
| HippoRAG2 | 81.20 | 0.826 | **0.841** |
| EraRAG | 55.00 | 0.558 | 0.592 |
| KAG | 64.33 | 0.257 | 0.588 |
| HyperGraphRAG | 68.00 | 0.697 | 0.577 |
| **PurifyGraphRAG** | **83.00** | **0.833** | 0.825 |

---

## 🤝 贡献

我们非常欢迎社区贡献！你可以通过以下方式参与 PurifyGraphRAG 的改进。

### 报告问题与建议

- 使用 [Issues](https://github.com/Rookie0086/PurifyGraphRAG/issues) 报告 Bug、提出新特性或改进建议。
- 提交 Issue 时请尽量提供运行环境（Python / Docker 版本）、复现步骤、期望与实际行为以及相关日志片段，以便更快定位问题。

### 提交代码

1. **Fork** 本仓库并创建特性分支：

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. 保持与现有代码一致的风格（PEP 8），并确保改动可通过最小验证：

   ```bash
   python -m src.PurifyGraphRAG
   ```

3. 提交 Pull Request 时，请清晰描述改动动机、实现方式与验证结果，并关联相关 Issue。
