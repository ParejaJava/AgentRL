# BGE 召回模型的 CUDA 依赖与显存适配

## 1. 调整目标

商品搜索和品类洞察都会使用 BGE-M3 生成向量，并使用
BGE-reranker-v2-m3 对候选结果重排。本次调整解决两个问题：

1. `uv sync --extra search` 必须稳定安装 CUDA 版 PyTorch，不能在下次同步时换回
   CPU 版。
2. 模型设备、计算精度和批大小必须由环境配置控制，适配当前约 8GB 的 NVIDIA
   显存，同时让商品检索、商品索引构建和品类洞察使用同一套策略。

## 2. 依赖固化

`pyproject.toml` 的 `search` extra 现在包含：

- `torch==2.11.0`
- `FlagEmbedding>=1.3,<2.0`
- `transformers>=4.44.2,<5.0`
- `faiss-cpu>=1.15,<2.0`

Windows 下的 `torch` 由 uv 的显式索引 `pytorch-cu130` 提供：

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu130", marker = "sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

这样做以后，普通 PyPI 仍负责其他依赖，只有 Windows 的 PyTorch 被定向到 CUDA
13.0 wheel。`uv.lock` 也记录了这套解析结果。

安装完整检索环境：

```powershell
uv sync --extra search --extra rag --extra dev
```

只运行商品搜索、不使用 OpenSearch 时，可省略 `rag`：

```powershell
uv sync --extra search --extra dev
```

## 3. 8GB 显存配置

当前 `.env` 和 `.env.example` 使用以下配置：

```dotenv
RETRIEVAL_DEVICE=cuda:0
RETRIEVAL_USE_FP16=true
RETRIEVAL_EMBEDDING_BATCH_SIZE=16
RETRIEVAL_RERANKER_BATCH_SIZE=8
```

各字段含义：

| 配置 | 作用 | 当前值的考虑 |
| --- | --- | --- |
| `RETRIEVAL_DEVICE` | 指定 FlagEmbedding 推理设备 | `cuda:0` 固定使用第一张显卡 |
| `RETRIEVAL_USE_FP16` | 以半精度加载和计算模型 | 明显降低两套模型常驻时的显存占用 |
| `RETRIEVAL_EMBEDDING_BATCH_SIZE` | 每批编码的文本数量 | 16 兼顾建索引速度和峰值显存 |
| `RETRIEVAL_RERANKER_BATCH_SIZE` | 每批交叉编码的查询-商品对数量 | 8 用于控制重排阶段的激活显存 |

`RETRIEVAL_DEVICE` 还接受 `auto`、`cpu`、`mps`、`cuda` 和 `cuda:<序号>`。
`auto` 会让 FlagEmbedding 根据 PyTorch 能力自动选择设备。生产或评测环境建议显式填写，
避免机器环境变化后静默切换执行设备。

## 4. 配置传递链

运行时的配置流如下：

```text
.env
  -> Settings.from_env()
  -> composition.py
  -> item_search/category_insight factory
  -> BGEEmbeddingEncoder / BGEReranker
  -> BGEM3FlagModel / FlagReranker
```

具体实现：

- `app/infrastructure/settings.py` 读取并校验四个 `RETRIEVAL_*` 字段。
- `app/composition.py` 给在线商品搜索和离线商品索引构建器传入相同配置。
- `app/infrastructure/retrieval/category_insight/factory.py` 让 OpenSearch 品类召回
  复用同一设备策略。
- `app/infrastructure/retrieval/item_search/bge.py` 在首次调用时才加载模型，并把
  `devices`、`use_fp16`、`batch_size` 和长度限制传给 FlagEmbedding。

模型仍然保持惰性加载：应用启动和依赖装配不会立刻占用显存。第一次 embedding
请求加载 BGE-M3，第一次 rerank 请求加载 reranker。

## 5. 为什么分别配置两个批大小

Embedding 与 reranker 的显存特征不同：

- Embedding 对单条文本做编码，批大小 16 通常能提高 GPU 吞吐。
- Reranker 同时编码查询和候选文档，输入更长，激活显存更高，因此批大小设为 8。

商品搜索的 `recall_k` 可能一次产生很多重排对，但适配器会按 reranker 批大小分批，
不会把全部候选一次性送进显卡。

如果出现 CUDA OOM，按以下顺序调整：

1. 把 `RETRIEVAL_RERANKER_BATCH_SIZE` 从 8 降到 4，再降到 2。
2. 把 `RETRIEVAL_EMBEDDING_BATCH_SIZE` 从 16 降到 8。
3. 缩短检索配置中的召回候选数或输入文本长度。
4. 仅用于排障时改为 `RETRIEVAL_DEVICE=cpu` 和
   `RETRIEVAL_USE_FP16=false`；CPU 可以运行，但速度会显著下降。

## 6. 验证命令

确认 CUDA PyTorch：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期关键结果：

```text
2.11.0+cu130 13.0 True
NVIDIA GeForce RTX 5060 ...
```

运行配置与召回相关测试：

```powershell
uv run pytest -q tests/test_retrieval_inference_config.py tests/test_item_search.py tests/test_category_insight.py tests/test_opensearch_category_insight.py
```

模型首次运行时可能需要几十秒加载权重。任务管理器或 `nvidia-smi` 中看到显存上升，
同时 Python 进程保持计算状态，属于正常现象。

## 7. 边界说明

- 本次没有把 BGE 模型对象放进领域层；设备与 CUDA 参数仍属于 infrastructure。
- `faiss-cpu` 只负责本地向量索引，不影响 BGE 在 GPU 上推理。
- OpenSearch 仍由 Docker 服务提供，Python 的 `opensearch-py` 只是客户端依赖。
- FP16 可能造成极小的浮点分数差异，因此离线评测应关注排序指标是否稳定，不应要求
  每个原始分数逐位相等。
