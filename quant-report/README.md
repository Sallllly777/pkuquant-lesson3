# quant-report：任务二研究归档

本文件夹归档第二个任务“数据初步清洗、因子探索与执行研究”的最终报告及此前三轮实验。研究对象为2026-09-23的50只股票Level-2数据。

**优先阅读：[最终因子研究报告](reports/final_research_20260923/最终因子研究报告.md)。** GitHub可以直接阅读Markdown及相对路径图表；[HTML版本](reports/final_research_20260923/最终因子研究报告.html)内嵌图表，下载后用浏览器打开。

## 报告与实验索引

| 轮次 | 报告 | 实验内容 |
|---|---|---|
| 第一轮 | [初步因子研究](reports/factor_research_20260923/初步因子研究.md) | 清洗质量、25个候选方向因子、预测时效、价差与延迟 |
| 第二轮 | [三条研究路线与执行成本](reports/execution_research_20260923/三条研究路线与执行成本.md) | 盘口下单择时、被动挂单成交与风险、盘口和订单流组合、报单延迟 |
| 第三轮／最终 | [最终因子研究报告](reports/final_research_20260923/最终因子研究报告.md) | 21,500个任务、三个预测目标、9组模型、11种执行规则、撤单延迟及后续实验标准 |

核心发现：排队信息能改善模拟成交预测，但成交后风险仍难以准确估计；更复杂的筛选与撤单规则没有自动改善执行。全部结果属于单日探索，其中未成交余量采用报价估值，不是实盘盈利证明。具体数值、失败实验及局限均见报告。

## 文件组织

```text
原仓库/
├── .git/                       # 沿用原仓库，quant-report 内没有第二个 .git
├── l2sim/                      # 第一任务的回放与模拟成交引擎
└── quant-report/
    ├── README.md               # 本索引与提交说明
    ├── .gitignore              # 放行实验结果，排除原始行情及大中间面板
    ├── .gitattributes          # 文本与二进制文件属性
    ├── run.py                  # 跨目录复现入口
    ├── verify.py               # 归档校验入口
    ├── archive_manifest.json   # 文件清单、大小、SHA256与本地保留项
    ├── requirements*.txt       # 研究依赖
    ├── research/               # 实验实现快照，Python文件保持原样
    ├── tests/                  # 42项引擎与研究测试快照
    └── reports/
        ├── factor_research_20260923/
        ├── execution_research_20260923/
        │   ├── arrival_1s/
        │   └── arrival_5s/
        └── final_research_20260923/
```

三个报告目录包含图表、参数、质量报告、统计结果、任务和逐笔模拟成交文件。Parquet为原实验输出格式，可用pandas或PyArrow读取；没有再次清洗或改写其数值。原始报告副本只增加归档阅读说明，并将图片链接改为相对路径；HTML内嵌图片。归档文本统一为LF换行，使Windows与其他系统检出后的哈希一致。

`research/README.md`与报告正文保留生成时的原工作区说明。**在此归档副本中运行实验，请以下文 `run.py` 命令为准**，避免调用仓库根目录同名研究包。

## Git包含范围

报告、图表、实验源码、测试、模型参数、小型汇总表、逐任务结果与模拟成交表都纳入本文件夹的Git提交范围。原始行情ZIP及清洗输入不在归档内。

以下四个可重建的大中间面板已复制到本地归档，默认被本文件夹的 `.gitignore` 排除，因此不会随普通 `git add quant-report` 上传：

| 本地保留文件 | 用途 |
|---|---|
| `reports/execution_research_20260923/scored_panel.parquet` | 带组合分数的全量5秒特征面板 |
| `reports/final_research_20260923/observed_states.parquet` | 订单状态与基础特征 |
| `reports/final_research_20260923/labeled_states.parquet` | 订单状态与未来标签 |
| `reports/final_research_20260923/predicted_states.parquet` | 状态、标签与所有模型预测 |

这样既保留本机完整实验现场，也避免把可重新生成的大型中间输入加入普通Git历史。克隆后的报告、参数、逐任务及逐笔结果可以直接阅读；重新训练需按下文重建这些面板。`archive_manifest.json`同时记录提交文件与本地保留项，缺少后者不会导致归档校验失败。

## 在原仓库中提交

本文件夹是普通仓库子目录，不要在其内部运行 `git init`。本次整理仅暂存此目录，没有创建提交或推送，原仓库已有的其他未提交工作保持原状态。

在原仓库根目录运行，先确认暂存区只包含自己希望提交的文件：

```powershell
git status --short
git diff --cached --stat
git add -- quant-report
git commit -m "Add quant-report research reports and experiments"
git push
```

当前机器的原仓库位于 `E:\pku-quant-3`；如果将文件夹复制到另一份已克隆的原仓库，就在那份仓库的根目录运行上述命令。报告图表使用相对路径，不依赖此盘符。HTML可下载本地阅读，GitHub页面默认阅读Markdown版本。

## 校验与复现

以下命令均在原仓库根目录执行。入口会优先加载本归档里的 `research/`，复用父目录的 `l2sim/`；新的报告写入本归档 `reports/`，原始ZIP保持只读，清洗缓存沿用父目录 `data/cleaned/20260923/`。

仅核对交付文件、相对链接和哈希，不需要安装研究依赖：

```powershell
python -B quant-report/verify.py
```

安装依赖并执行归档内42项测试，不需要行情ZIP：

```powershell
python -m pip install -r quant-report/requirements-research.txt
python -B quant-report/run.py test
```

完整重现时，需要自行准备与原研究一致的 `sse50_20260923.zip`，放在父仓库根目录。原包SHA256：`18c0e124a9c68eb2f77b965ca1f29c93bb30339d7db45245e1af3c0420c6c947`。

```powershell
python -B quant-report/run.py pipeline all
python -B quant-report/run.py execution_study all
python -B quant-report/run.py execution_study latency
python -B quant-report/run.py execution_report
python -B quant-report/run.py final_study all
python -B quant-report/run.py final_report
```

复现会覆盖归档中的同名派生输出，属于新的运行，不再保证与交付清单逐字节一致。`manifest*.json`是原实验的运行证据，`archive_manifest.json`是本次打包的文件证据，两者用途不同。该文件夹依赖原仓库的模拟引擎，不是一个独立交易系统。
