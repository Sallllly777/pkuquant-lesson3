"""Readable explanation of the three execution research experiments."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .execution_study import OUT
from .report import table


def write_report():
    summary = json.loads((OUT / "execution_summary.json").read_text(encoding="utf-8"))
    group = pd.read_parquet(OUT / "passive_groups.parquet")
    policies = pd.read_parquet(OUT / "timing_policies.parquet").set_index("policy")
    ic = pd.read_parquet(OUT / "combination_ic.parquet")
    cross = pd.read_parquet(OUT / "combination_cross_ic.parquet")
    cells = pd.read_parquet(OUT / "book_flow_cells.parquet")
    latency = pd.read_parquet(OUT / "latency_cost_decomposition.parquet")
    regimes = group[group.grouping == "obi_regime"].set_index("group")
    all_ = group[group.grouping == "ALL"].iloc[0]
    labels = {"score_book": "一档盘口", "score_flow": "近5秒主动成交方向", "score_book_flow": "盘口＋主动成交",
              "score_book_flow_depth": "再加多档深度", "score_book_flow_depth_ofi": "再加OFI"}
    pm = ic[ic.split == "PM"].pivot(index="factor", columns="target", values="ic")
    combo_table = table(["模型", "未来5秒IC", "未来30秒IC", "未来120秒IC"],
                        [[name] + [f'{pm.loc[f, "ret_" + str(h) + "s"]:.3f}' for h in [5, 30, 120]] for f, name in labels.items()])
    policy_names = {"wait_30s": "所有任务都等30秒", "obi_timing_5s": "盘口顺向立即成交，否则等5秒",
                    "obi_timing_30s": "盘口顺向立即成交，否则等30秒", "combo_timing_30s": "组合预测顺向立即成交，否则等30秒",
                    "passive30_then_quote_remainder": "全部挂最优价30秒，余量按届时报价估值",
                    "combo_aggressive_else_passive": "组合顺向立即成交，否则挂单30秒并估值余量"}
    policy_table = table(["固定交易任务的执行方式", "比立即成交改善/bp", "改善为正的股票比例"],
                         [[name, f'{policies.loc[key, "mean_saving_bps"]:.3f}', f'{policies.loc[key, "positive_stock_fraction"]:.0%}'] for key, name in policy_names.items()])
    regime_names = {"favorable": "顺初始盘口方向", "neutral": "盘口中性", "adverse": "逆初始盘口方向"}
    passive_table = table(["下单时状态", "30秒内有成交", "队列前方股数中位数", "成交后30秒方向收益/bp", "成交价相对30秒后中间价/bp"],
                          [[name, f'{regimes.loc[key, "any_fill"]:.1%}', f'{regimes.loc[key, "median_queue_ahead"]:,.0f}',
                            f'{regimes.loc[key, "postfill_drift_30_bps"]:.3f}', f'{regimes.loc[key, "markout_30_bps"]:.3f}'] for key, name in regime_names.items()])
    lat = latency[latency.factor == "obi1"].sort_values("delay_seconds")
    latency_table = table(["等多久才开始持有", "随后30秒有利中间价变化/bp", "两端价差损耗/bp", "按报价进出、费用前/bp"],
                          [[str(int(r.delay_seconds)) + "秒", f'{r.mid_bps:.3f}', f'{r.spread_drag_bps:.3f}', f'{r.quote_bps:.3f}'] for r in lat.itertuples()])
    fixed_exit_table = table(["延迟入场", "固定在信号后30秒退出的中间价变化/bp", "入场报价恶化/bp"],
                             [[str(int(r.delay_seconds)) + "秒", f'{r.fixed_exit30_mid_bps:.3f}', f'{r.entry_slippage_bps:.3f}'] for r in lat.itertuples()])
    state_labels = {"negative": "偏卖", "neutral": "中性", "positive": "偏买"}
    cell_table = table(["盘口", "主动成交", "样本数", "未来30秒均值/bp", "逐股等权均值/bp", "平均价差/bp"],
                       [[state_labels[r.book_state], state_labels[r.flow_state], f'{r.n:,}', f'{r.mid_30_bps:.3f}',
                         f'{r.stock_equal_mid_30_bps:.3f}', f'{r.spread_bps:.3f}'] for r in cells.itertuples()])
    arrival_rows = []
    for delay in [0, 1, 5]:
        folder = OUT if delay == 0 else OUT / f"arrival_{delay}s"
        s = json.loads((folder / "execution_summary.json").read_text(encoding="utf-8"))
        g = pd.read_parquet(folder / "passive_groups.parquet")
        a = g[g.grouping == "ALL"].iloc[0]
        pol = pd.read_parquet(folder / "timing_policies.parquet").set_index("policy")
        arrival_rows.append({"delay": delay, "any_fill": a.any_fill, "full_fill": a.full_fill,
                             "taker_task_fraction": s.get("taker_task_fraction", 0), "markout30": a.markout_30_bps,
                             "postfill30": a.postfill_drift_30_bps,
                             "passive_saving": pol.loc["passive30_then_quote_remainder", "mean_saving_bps"]})
    arrivals = pd.DataFrame(arrival_rows)
    arrivals.to_parquet(OUT / "arrival_delay_comparison.parquet", index=False)
    arrival_table = table(["原价限价单到达延迟", "有成交比例", "到达即发生主动成交的任务比例", "成交价相对成交后30秒中间价/bp", "含未成交余量估值的执行改善/bp"],
                          [[str(int(r.delay)) + "秒", f'{r.any_fill:.1%}', f'{r.taker_task_fraction:.1%}', f'{r.markout30:.3f}', f'{r.passive_saving:.3f}'] for r in arrivals.itertuples()])
    plot(cells, regimes, lat, policies)
    def img(name): return (OUT / name).resolve().as_posix()
    text = f"""# 盘口择时、被动成交风险与订单流组合：第二轮研究

本轮实际运行了三类实验：组合预测、固定交易任务的执行比较、逐笔排队成交与延迟。核心发现是：**预测上涨、买得更便宜、挂单能够成交，是三个不同的目标。** 一个因子可以预测方向，却不够支付主动往返价差；也可以帮助一笔本来就要做的交易节约成本。

原始数据仍为2026-09-23、50只股票，ZIP哈希核对不变。本日午后在第一轮已被看过，因此第二轮属于进一步探索，不能再称全新样本外检验。

## 1. 为什么盘口与订单流有信息

一档挂单不平衡为 `OBI=(买一量−卖一量)/(买一量+卖一量)`。例如买一9000股、卖一1000股，OBI=0.8。在补单、撤单、主动买卖相近的简化条件下，较薄的卖一更容易先被消耗，下一次中间价向上调整的机会较大。这是概率机制，不是厚买单必然托住股价；厚单可以撤走，对面的卖盘也可以持续补进。

近5秒主动成交不平衡衡量的是刚刚实际消耗流动性的方向。它与盘口的区别在于：盘口记录当前可见存量，成交记录已经发生的消耗。固定价位的队列满足“剩余量变化＝新增挂单−撤单−成交消耗”。只看存量，可能漏掉快速消耗；只看成交，可能漏掉对手补单。两者联合才有机会区分短暂成交和持续的供需失衡。

需要避免一个重复因子：静态挂单量加权中点满足 `加权中点−普通中点=(卖一−买一)×OBI/2`。所以它本质上是价差缩放后的OBI，不自动产生独立信息，也不是已经估计出来的真实均衡价格。

本数据的下午条件分组如下。偏买/偏卖的阈值固定为±0.5，等价于对应一侧数量超过另一侧3倍；股票量和成交量分别按其自身定义计算。

{cell_table}

盘口与主动成交同时偏买时，未来30秒平均上涨约0.925bp；盘口偏买但主动成交偏卖时，只剩约0.103bp。这与“成交流是否支持盘口状态”相符。两者同时偏卖时约−1.307bp。各组的股票组成、价差和波动并不相同，因此同时给出逐股等权均值；这些分组不能单独证明因果关系。

更高均值也不等于更高无条件上涨概率：很多时段中间价不动，不同组的波动幅度不同。应同时看不变比例、方向概率、变化幅度和成本。

![盘口与订单流条件均值]({img('book_flow_cells.png')})

## 2. 组合信息：有增量，但收益取决于目标窗口

模型仅用上午拟合：逐股标准化特征和未来30秒收益，再做固定正则强度0.001的线性拟合。下午不调整系数。比较一档盘口、主动成交、多档深度和OFI的逐步加入，不进行参数搜索。组合分数表示相对上午常态的方向强弱，不是直接以基点计价的预期收益；执行规则以分数0为固定阈值。

{combo_table}

加入主动成交，30秒平均逐股IC从0.188提升到0.196；再加入多档深度后为0.204。但只有26/50只股票在两因子组合上超过盘口基线，加入多档后也只有27/50只，不能说普遍改善。再加OFI的平均增量仅约0.0003。

未来5秒的IC反而由一档盘口约0.250降低到多档组合约0.220，未来120秒则从约0.142升至0.180。模型针对30秒训练；快信号和慢状态混合后，可能改善较慢目标，同时稀释最短时效的信息。

横截面排名也不完全同步改善：标准化分数的下午30秒横截面IC，一档约0.158、盘口＋成交约0.166、再加多档约0.150。按股预测时间变化与在股票之间排名，是不同任务；这里的标准化分数也不与第一轮原始OBI的横截面数值直接混比。

## 3. 盘口信号辅助一笔既定交易择时

这项实验假定本来就有一笔买入任务或卖出任务，研究怎样执行，而不是用预测信号决定先买再卖。每只股票从13:05至14:49每分钟取一个时点，分别模拟买入和使用已有可卖库存的卖出：主板100股、科创板200股，共10,500个小订单任务。

每个任务独立重置资金和底仓，盘口继续按真实逐笔流前进。比较立即按对手盘可见深度成交、等待后成交、以当前最优本方报价挂单30秒。买卖任务是独立对照，不是一个T+0套利组合。这些固定时点、买卖均衡的任务是实验设计，不是观测到的真实策略发单分布；换成实际策略的下单时点和方向后必须重新检验。

以“同一时刻立即完成同一笔任务的价格”为基准，正值代表买得更低或卖得更高：

{policy_table}

仅仅所有订单都等待30秒，平均几乎没有改善。根据盘口或组合判断紧迫性再决定是否等待，改善约0.21—0.27bp；组合择时在43/50只股票上的改善为正，单笔改善的中位数仍为0，说明效果不是每次都出现。

这里“顺向”指买入任务遇到看涨信号，或卖出任务遇到看跌信号：预期再等对自己更不利，因此先完成；反向时允许等待。它利用的是**等待成本的差异**。即便0.27bp不足以支付一次往返价差，也可能改善本来就必须发生的一笔交易。

被动与混合方式的约0.83/0.92bp改善，含未成交量在截止时刻按对手报价估值。它没有丢弃未成交订单，但也不是所有剩余量都真的发单成交。零延迟样本有27个部分成交任务，占0.257%；这些剩余量不满足本模拟器最低申报量条件，需要与后续任务合并、调整数量或继续等待。本轮没有把它们冒充成可直接重新申报的完整执行策略。

对同一笔必做任务，双方相同的佣金通常会在相对成本比较中抵消；拆成多张订单、最低收费、撤改与未完成风险可能重新产生差异。这与“产生一个新往返交易必须赚回全部交易成本”是两种经济问题。

![固定任务执行改善]({img('timing_comparison.png')})

## 4. 被动挂单：初始看涨不等于成交后还会涨

零延迟时，30秒内有成交的任务占{all_.any_fill:.1%}，完全成交占{all_.full_fill:.1%}，平均成交比例{all_.fill_fraction:.1%}。已成交订单的平均成交等待约{all_.wait_seconds:.2f}秒。没有用“触及限价”代替成交，而是按前方队列、历史成交和新增穿价挂单逐笔推演。

为同时比较买卖，所有方向先转成“对订单是否有利”：买入后上涨为正、卖出后下跌为正。按下单时的 `订单方向×OBI` 分组，大于0.5为顺初始盘口，小于−0.5为逆初始盘口。

{passive_table}

本日最值得注意的是：**顺初始盘口的挂单更难成交，而且一旦成交，之后表现反而更差。** 顺向组队列前方中位数44,500股，逆向组只有3,940股；相应有成交比例约29.6%与69.6%。顺向组成交后30秒方向收益约−0.876bp，逆向组约+0.121bp。

一种与这些观察相符的机制是：看涨时，买一厚，跟随买入要排很长队。行情按预期上涨时，它常常买不到；真正买到的样本，可能恰好是后续强卖压把长队吃掉、原先信号已经失效的时刻。初始预测针对所有时点，而成交后结果只针对“成功穿过队列的那部分样本”，并且计时起点已经变成成交时刻。

这也说明不能套用“成交率越高，成交后必然越差”的单一规律：本日按初始OBI分组，反而是长队、低成交率的一组条件结果更差。需要同时考虑初始信号、队列位置、等待期间的新信息和成交条件；观察到的关系仍可能受股票与行情状态差异影响。

真实风险目标应写成 `E[成交后收益 | 当前状态, 已经成交]`，而不是只优化 `E[未来价格变化 | 当前状态]`。挂单等待中应重算信号；但撤单再挂也会失去优先级，因此不能认为无限频繁撤改总是有利。

![排队与成交后风险]({img('passive_selection.png')})

零延迟下，已成交订单相对成交时参考中间价的平均优势仅约0.323bp，之后30秒平均不利变化约0.288bp，剩下约0.035bp的中间价标记优势，尚未扣费、也没有退出成交。它不是挂单赚到半个初始价差的证据。部分成交发生时，参考盘口已经移动，原来看到的价差优势早就改变了。

约15%的成交记录通过“后续新增剩余挂单穿越模拟限价”推断，其余通过成交记录和队列推断。参考盘口并未因模拟订单改变，模型没有再生后续订单行为。真实自有订单如果改变最优报价、成交顺序或对手反应，结果可能不同；这些模型来源已经逐笔保存。

## 5. 交易成本究竟吞掉了什么

以买入后卖出的诊断为例，近似有：

```text
按买卖报价进出的变化
≈ 中间价变化 − 入场半价差 − 退出半价差

进一步扣除：佣金、税费、超出最优档的冲击、其他滑点。
```

两端半价差合计约一个完整价差，不能误写成扣两次完整价差。该式是成本分解，不提供普通股票无底仓日内往返能力。

一档盘口强信号的27,240个下午观察点：

{latency_table}

零延迟的0.781bp有利中间价变化，被约6.778bp的两端价差损耗覆盖，结果约−5.997bp。这不是因为手续费把一个原本很赚钱的策略变亏；**在手续费之前，价差已经大于预测幅度。** 若再假设单边费用1bp，往返再减少2bp；单边3bp则再减少6bp。这里只是敏感性参数，不假设你的实际券商费率。

最小申报量会放大小额交易的固定收费：例如一笔2000元的订单，若最低收费为5元，单边就是25bp。按本轮任务金额计算，“每笔固定5元”这一假设对应的单边费用中位数约{summary['min5_commission_median_bps']:.2f}bp。机构或其他账户的收费可能完全不同，因此不能把此情景当成真实费率。

增加订单金额虽可能摊薄固定费用，但也可能吃到更多档位，增加冲击和排队时间。本轮只有100/200股的小单结果，尚未证明可扩大容量。

## 6. 执行速度的两种影响

第一种是信号变旧。上述延迟实验每次都保持随后30秒的观察长度：从立即开始的0.781bp降到晚1秒的0.511bp、晚5秒的0.274bp。这里窗口终点也随之移动，不能直接把差值说成“前5秒已实现的全部收益”，更不能估成精确半衰期。

因此另外固定在原信号后30秒退出，单独观察晚入场损失：

{fixed_exit_table}

晚5秒按对手报价进场，平均入场价格已经恶化约0.547bp。更快入场能减少这项损失，但无法消除约6.8bp的价差；单靠加速不足以使上述主动往返诊断转正。

第二种是订单在队列和盘口中的位置变化。补充实验保持信号时刻的原始限价，在0/1/5秒之后才送达，并从实际到达时开始等待30秒再撤单；取消本身仍设为即时。行情变动时，原本想挂买一的价格可能已经高于到达时卖一，于是限价单会立即主动成交。原信号、旧限价、晚到达之间的区别被保留下来，没有把晚到的单自动换成新报价。

{arrival_table}

这组实验说明应同时看成交方式、成交后质量及未成交的代价，不能只看成交率。由于等待截止时刻也随到达延迟向后移动，它包含市场路径变化、排队优先级变化和旧限价影响，不是孤立估计某一个通道的网络耗时。

真实系统还有行情接收、特征计算、决策、报单到达、撤单到达等多个环节。本轮没有实际网络时间戳，也没有估计毫秒级真实延迟分布；1/5秒实验是敏感性分析。撤单延迟导致继续暴露于旧报价、撤后重挂丢失队列位置，属于下一步需要单独加入的机制。

![信号陈旧与价差]({img('latency_cost.png')})

## 7. 对下一轮因子研究的具体启发

1. **先选目标再选因子。** 方向模型用未来中间价；执行模型用相同任务的实现成本；挂单模型同时预测完成概率、等待时间和成交后不利变化。三个目标分别验收，不用一个IC包办。
2. **同时估计方向和幅度。** 胜率、Rank IC之外，要看预期基点、价格不变比例、价差及订单大小。预测正确但仅移动半个跳，可能根本无法覆盖主动进出。
3. **只接受有增量的组合。** 控制原有OBI、价差与波动后再评估新因子；按股票、时段、方向分组。平均IC提升而多数股票无改善、横截面排名变差时，不应宣布全面优胜。
4. **按时效拆模型。** 一档盘口用于最短时段，成交流和多档状态用于较慢窗口；评价“实际可到达时间之后”的收益，不用计算完成前已经发生的价格变化奖励模型。
5. **把排队当成特征。** 前方量、对手成交消耗、撤单推进、等待时长、已成交比例，应进入成交概率和风险模型。到价距离相同不代表成交机会相同。
6. **不要只在成交样本挑结果。** 报告所有委托的完成率、未完成量、截止成本、部分成交处理；成交后风险另外条件化。否则可能只留下看似成功的低成本成交。
7. **动态刷新风险。** 入场时信号好，几秒后可能已经坏。需要检测订单流转向和补单消失，但把撤改延迟、失去队列优先级和报单约束一起计入。
8. **先验证执行改善，再验证收益。** 本日约0.27bp的择时改善有可能服务于原有交易任务；要把它变成独立收益策略，还需额外支付周转、资金与库存风险，不能直接换算成年化。
9. **补多日并锁定方案。** 当前只有一个交易日，而且午后已被反复观察。固定特征、窗口、模型、费用与订单规则后，再做跨日期检验；按日或较长时间块评估不确定性，不能把31,500个情景任务当独立实验。

建议下一阶段先做三输出的简单模型：`预期价格变化`、`期限内成交概率`、`成交后不利变化`，把这三项输入现有回放系统的“立即成交/等待/挂单/撤单”选择。暂不需要用更复杂模型掩盖执行假设。

## 方法边界、文件与复现

预测部分复用129,000行研究样本，上午拟合、下午探索；执行部分每个延迟场景10,500个任务，三种场景共31,500。相同股票、时点和历史路径重复使用，不能视为独立样本增加了三倍。卖出使用假定已有可卖库存，不绕过T+1，也不建立无库存做空。

原始ZIP与第一轮清洗文件不修改。源码为 `research/execution_study.py`、`research/execution_report.py`，输出均在本报告目录。`scored_panel.parquet`、`combination_ic.parquet`、`book_flow_cells.parquet` 保存组合分析；`passive_tasks.parquet`、`passive_fills.parquet` 保存逐任务与逐笔模拟成交；`execution_enriched.parquet` 保存合并结果；`arrival_1s/`、`arrival_5s/` 保存延迟实验。所有生成数据仍由Git忽略。

在项目根目录、具备 `requirements-research.txt` 依赖的Python环境运行：

```powershell
python -B -m research.execution_study all
python -B -m research.execution_study latency
python -B -m research.execution_report
```

`all` 包含零延迟实验；`latency` 包含1/5秒延迟实验；最后生成本报告与图表。规则、资金股份、无未来信息、队列匹配、部分成交剩余量、延迟限价单转为主动成交等测试共35项通过。

## 研究出处

队列不平衡预测下一次中间价变化的思路参考[Gould与Bonart的研究](https://arxiv.org/abs/1512.03492)；本项目数字均来自本地数据重新计算，并不直接迁移其市场结论。

OFI与短期价格冲击的联系参考[Cont、Kukanov与Stoikov](https://arxiv.org/abs/1011.6402)。本轮使用历史窗口预测未来窗口，避免把同期冲击解释当成未来预测收益。

限价订单的选择偏差、流动性不平衡和延迟之间的关系可参考[Lehalle与Mounjid](https://arxiv.org/abs/1610.00261)。成交概率与成交后收益需要联合考虑，也见[Albers等人的研究](https://arxiv.org/abs/2502.18625)；后者研究的是加密资产，本文只借鉴问题设定，不假定其结论必然适用于A股。
"""
    path = OUT / "三条研究路线与执行成本.md"
    path.write_text(text, encoding="utf-8")
    print(path)


def plot(cells, regimes, latency, policies):
    order = ["negative", "neutral", "positive"]
    values = cells.pivot(index="book_state", columns="flow_state", values="mid_30_bps").loc[order, order]
    fig, ax = plt.subplots(figsize=(7, 5))
    limit = np.abs(values.to_numpy()).max()
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(3), ["Sell-heavy", "Neutral", "Buy-heavy"])
    ax.set_yticks(range(3), ["Ask-heavy", "Neutral", "Bid-heavy"])
    ax.set_xlabel("Last 5s aggressive trade volume"); ax.set_ylabel("Level-1 resting quantity")
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{values.iloc[i,j]:+.3f} bp", ha="center", va="center", fontsize=12)
    ax.set_title("Afternoon: average next-30s mid-price change")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(OUT / "book_flow_cells.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    z = regimes.loc[["adverse", "neutral", "favorable"]]
    names = ["Against initial\nbook signal", "Neutral", "With initial\nbook signal"]
    axes[0].bar(names, z.any_fill * 100, color="#276E8F")
    for i, v in enumerate(z.any_fill): axes[0].text(i, v * 100 + 1, f"{v:.1%}", ha="center")
    axes[0].set_ylim(0, 85); axes[0].set_ylabel("Any-fill probability within 30s (%)")
    axes[1].bar(names, z.postfill_drift_30_bps, color=["#398F78", "#CC8A47", "#BA4D45"])
    for i, v in enumerate(z.postfill_drift_30_bps): axes[1].text(i, v + (.03 if v > 0 else -.06), f"{v:+.3f}", ha="center")
    axes[1].axhline(0, color="grey", linewidth=.6); axes[1].set_ylabel("Signed mid-price change 30s AFTER FILL (bp)")
    axes[1].set_ylim(-1.08, .3)
    fig.suptitle("Passive orders: initial signal vs conditional execution outcome")
    fig.tight_layout(); fig.savefig(OUT / "passive_selection.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(latency.delay_seconds, latency.mid_bps, "o-", label="Signed mid-price change; next 30s")
    ax.plot(latency.delay_seconds, latency.quote_bps, "s-", label="Bid/ask round-trip markout; no fees")
    for r in latency.itertuples():
        ax.annotate(f"{r.mid_bps:.3f}", (r.delay_seconds, r.mid_bps), xytext=(0, 8), textcoords="offset points", ha="center")
        ax.annotate(f"{r.quote_bps:.3f}", (r.delay_seconds, r.quote_bps), xytext=(0, -15), textcoords="offset points", ha="center")
    ax.set_xticks([0, 1, 5]); ax.set_xlabel("Entry delay from original signal (seconds)"); ax.set_ylabel("Basis points")
    ax.axhline(0, color="grey", linewidth=.6); ax.set_ylim(-7.5, 1.7)
    ax.set_title("OBI1 tail signals: speed helps, spread still dominates")
    ax.legend(loc="center right"); fig.tight_layout(); fig.savefig(OUT / "latency_cost.png", dpi=160); plt.close(fig)
    names = ["wait_30s", "obi_timing_30s", "combo_timing_30s", "passive30_then_quote_remainder", "combo_aggressive_else_passive"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals = policies.loc[names, "mean_saving_bps"]
    ax.barh(range(5), vals, color=["#A6ADB4", "#276E8F", "#398F78", "#CC8A47", "#CC8A47"])
    ax.set_yticks(range(5), ["Wait 30s for all", "OBI conditional timing", "Book + flow conditional timing", "Passive; value remainder at deadline*", "Immediate or passive by signal*"])
    for i, v in enumerate(vals): ax.text(max(v, 0)+.025, i, f"{v:+.3f}", va="center")
    ax.set_xlim(-.1, 1.15); ax.axvline(0, color="grey", linewidth=.6)
    ax.set_xlabel("Improvement over immediate execution (bp); not portfolio return")
    ax.set_title("10,500 mandatory small-order tasks, before differential fees")
    fig.text(.5, .01, "*Residual valuation is not an actual fallback submission; zero latency and no market impact.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .035, 1, 1]); fig.savefig(OUT / "timing_comparison.png", dpi=160); plt.close(fig)


if __name__ == "__main__": write_report()
