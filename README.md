# FusionNeo

FusionNeo 是一个受 FusOn-pLM 启发的融合断点新抗原建模项目。它在 ESM-2
基础上继续进行 masked language modeling，并提高融合断点附近残基的掩蔽概率，
随后导出 junction peptide、断点上下文和完整局部上下文的 embedding。

本仓库实现的是研究原型，不是临床诊断软件。

## 设计原则

- 优先使用实验或转录本注释得到的精确蛋白断点。
- 缺少精确坐标时，才用 head/tail 野生型蛋白局部比对推断断点。
- MHC-I 候选肽默认生成 8–14mer，并保证肽跨越断点。
- 模型读取断点两侧较长上下文，而不是只读取 8–11 个氨基酸。
- 训练、验证和测试按 `head::tail` 融合基因对划分，避免同一融合泄漏。
- 同时生成融合蛋白非断点肽和 head/tail 野生型肽作为背景。
- 支持普通固定掩蔽、FusOn-pLM 式余弦均匀掩蔽和断点偏置余弦掩蔽。

坐标全部使用 Python 约定：从 0 开始，区间右端不包含。`breakpoint=100`
表示断点位于 `sequence[99]` 与 `sequence[100]` 之间。

## 1. 安装

建议使用 Python 3.10 或 3.11。

```bash
git clone https://github.com/YOUR_NAME/FusionNeo.git
cd FusionNeo
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -e .
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
```

有 NVIDIA GPU 时，请先按照 PyTorch 官方说明安装与本机 CUDA 匹配的 PyTorch，
再运行 `pip install -e .`。

## 2. 下载 FusOn-DB

```bash
fusionneo-download --output data/raw/fuson_db
```

等价的 Python 调用是：

```python
from datasets import load_dataset

ds = load_dataset("ChatterjeeLab/FusOn-DB")
```

下载命令会把 Hugging Face 数据保存成可重复读取的本地 Dataset。

## 3. 准备 head/tail 野生型蛋白

推荐方案是根据你自己的融合转录本注释，准备与样本一致的蛋白 isoform。FASTA
记录 ID 必须是基因名，例如：

```text
>EML4
MSV...
>ALK
MGA...
```

如果暂时没有 isoform 数据，可以下载 reviewed human UniProt 蛋白作为初始参考：

```bash
fusionneo-fetch-uniprot \
  --dataset data/raw/fuson_db \
  --output-fasta data/reference/uniprot_human_fusion_genes.fasta
```

该步骤会缓存 FASTA、UniProt 表格和未找到的基因列表。自动选择的 UniProt
序列不一定与每条融合记录的真实 isoform 一致，因此后续必须查看断点置信度和
失败表。

## 4. 准备断点、上下文和候选肽

### 推荐：提供精确断点

建立 `data/reference/breakpoints.tsv`：

```text
seq_id	breakpoint
seq1	321
seq2	417
```

然后运行：

```bash
fusionneo-prepare \
  --dataset data/raw/fuson_db \
  --reference-fasta data/reference/uniprot_human_fusion_genes.fasta \
  --breakpoint-table data/reference/breakpoints.tsv \
  --output data/processed
```

表中存在的 `seq_id` 使用显式断点；其余记录使用 head/tail alignment 推断。

### 只有蛋白序列时

```bash
fusionneo-prepare \
  --dataset data/raw/fuson_db \
  --reference-fasta data/reference/uniprot_human_fusion_genes.fasta \
  --output data/processed \
  --flank 50 \
  --lengths 8 9 10 11 12 13 14 \
  --min-confidence medium
```

如果研究目标仅限癌症融合，可额外添加 `--cancer-only`。默认保留 FusOn-DB
中的全部记录，便于把 non-cancer fusion 用作对照或单独实验。

主要输出：

- `contexts.parquet`：MLM 使用的断点上下文。
- `peptides.parquet`：跨断点肽、融合非断点肽和野生型背景肽。
- `failures.tsv`：缺失参考序列、低置信度或无法定位的记录。
- `summary.json`：样本数、切分和置信度统计。

先用少量数据检查流程：

```bash
fusionneo-prepare ... --limit 100
```

## 5. 训练三个消融模型

默认配置在 `configs/base.yaml`。默认模型是约 8M 参数的
`facebook/esm2_t6_8M_UR50D`，适合先验证代码。正式实验可以切换到 35M、
150M 或 650M 模型；模型越大，显存需求越高。

### A. 标准 15% 随机掩蔽

```bash
fusionneo-train \
  --config configs/base.yaml \
  --strategy random_fixed \
  --output outputs/esm2-random-15
```

### B. FusOn-pLM 式 15%–40% 余弦均匀掩蔽

```bash
fusionneo-train \
  --config configs/base.yaml \
  --strategy cosine_uniform \
  --output outputs/esm2-cosine-uniform
```

### C. FusionNeo 断点偏置余弦掩蔽

```bash
fusionneo-train \
  --config configs/base.yaml \
  --strategy cosine_junction \
  --output outputs/esm2-cosine-junction
```

断点偏置权重为：

```text
w(i) = 1 + boost * exp(-distance(i, breakpoint) / tau)
```

总掩蔽率仍按 15%–40% 余弦变化，但断点附近残基被选中的概率更高。

首次运行建议：

```bash
fusionneo-train --config configs/base.yaml --smoke-test
```

注意：三种模型必须使用相同的数据切分、模型大小、学习率和随机种子，才能把
差异归因于掩蔽策略。

## 6. 导出 FusionNeo embedding

```bash
fusionneo-embed \
  --model outputs/esm2-cosine-junction \
  --peptides data/processed/peptides.parquet \
  --output outputs/fusionneo-embeddings
```

输出包括：

- `peptide_embeddings.npy`
- `junction_embeddings.npy`
- `context_embeddings.npy`
- `metadata.parquet`

非断点和野生型背景没有 junction，因而对应的 junction embedding 为 `NaN`；
peptide embedding 和 context embedding 始终可用。

运行简单的 leakage-safe linear probe：

```bash
python scripts/benchmark_embeddings.py \
  --embeddings outputs/fusionneo-embeddings/peptide_embeddings.npy \
  --metadata outputs/fusionneo-embeddings/metadata.parquet \
  --output outputs/junction_linear_probe.json
```

## 7. SysteMHC、IEDB 或 CEDAR benchmark

不同数据库的导出字段会变化，因此仓库采用显式列映射，不把某一版本的列名
写死。首先把下载表标准化。例如，假设输入文件包含 `Sequence`、`Assay result`
和 `HLA`：

```bash
python scripts/prepare_external_benchmark.py \
  --input data/external/iedb_export.csv \
  --output data/external/iedb_normalized.csv \
  --peptide-column Sequence \
  --label-column "Assay result" \
  --positive-value Positive \
  --allele-column HLA
```

如果原始数据包含研究、患者或蛋白来源列，应使用 `--group-column` 指定它，
防止同一实验来源同时进入训练集和测试集：

```bash
python scripts/prepare_external_benchmark.py ... --group-column Study_ID
```

导出外部肽 embedding：

```bash
python scripts/embed_external_peptides.py \
  --model outputs/esm2-cosine-junction \
  --input data/external/iedb_normalized.csv \
  --output-dir outputs/iedb-embeddings
```

然后进行线性探针评估：

```bash
python scripts/benchmark_embeddings.py \
  --embeddings outputs/iedb-embeddings/peptide_embeddings.npy \
  --metadata outputs/iedb-embeddings/metadata.csv \
  --output outputs/iedb-linear-probe.json
```

呈递和免疫原性必须分开报告：

- HLA ligand/呈递任务：SysteMHC 或 IEDB ligand 数据。
- T-cell immunogenicity：IEDB/CEDAR 功能实验数据。
- “未检测到”不能自动当成可靠的生物学阴性。
- 应按 HLA allele 分层报告 AUROC、AUPRC，并报告每个 allele 的样本量。

当前线性探针只检验 embedding 是否携带有用信息。正式的 HLA
presentation 模型还应把 HLA pseudo-sequence 作为额外输入，不能只输入 peptide。

## 8. 与 NetMHCpan、ESM-2 和 ProtT5 比较

建议固定同一测试集，比较：

1. BLOSUM 或氨基酸组成特征。
2. 原始 ESM-2 peptide embedding。
3. ProtT5 peptide embedding。
4. `random_fixed` 模型。
5. `cosine_uniform` 模型。
6. `cosine_junction` 模型。
7. NetMHCpan EL score/rank（仅用于 HLA presentation 任务）。

NetMHCpan 是任务预测器，而 ESM-2/ProtT5/FusionNeo 是表征模型。公平比较时，
应给各种 embedding 使用相同的下游分类器；NetMHCpan 作为独立预测基线报告。

## 9. 推荐正式实验

- 主切分：按 `fusiongenes` 分组。
- 更严格切分：按 breakpoint cluster 或序列相似性聚类后分组。
- 外部测试：完全未参与 FusOn-DB 训练的融合新抗原集合。
- 消融：掩蔽策略、context 长度、junction boost、模型大小。
- 指标：AUROC、AUPRC、MCC、校准误差和置信区间。
- 重复：至少 3–5 个随机种子。

不要先随机拆分 peptide 再训练；同一融合产生的高度相似肽会造成严重泄漏。

## 10. 测试

```bash
pip install -e ".[dev]"
pytest -q
```

GitHub Actions 会在每次 push 和 pull request 时自动运行测试。

## 已知限制

- UniProt 基因级参考不保证匹配 FusOn-DB 中每条记录的真实转录本 isoform。
- 蛋白比对推断出的断点应视为候选区间，不应替代基因组/转录本断点。
- FusOn-DB 包含 non-cancer 记录；是否纳入训练应根据研究问题决定。
- 当前仓库未捆绑受许可限制的 NetMHCpan，也不重新分发 IEDB/CEDAR/SysteMHC 数据。
- 模型输出不能直接解释为临床可用的新抗原证据。
