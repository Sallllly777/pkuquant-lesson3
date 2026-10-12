# 本地验证输出

此目录存放运行生成的审计、模拟成交、快照时钟诊断和界面截图。为避免提交行情样本及不断变化的运行结果，Git 仅跟踪本说明文件。

在项目根目录准备好本地数据 `sse50_20260923.zip` 后，可重新生成结果：

```powershell
python -B -m l2sim demo --output reports/demo.json
python -B -m l2sim audit --all --output reports/audit_all.json
python -B scripts/snapshot_clock.py
```

原始数据需自行取得并保留在本地，仓库不包含数据下载或对外分发功能。详细验证结论见 `docs/研究与验证.md`。
