# Materon TF2.9 Multi-Backbone Engine

统一的 TensorFlow 2.9 / kgcnn 2.1.0 核心引擎，覆盖 CGCNN、MEGNet 与 ElemNet 的 Materon 接入、权重配对、恒等性验收和解释导出。

- 冻结来源：`Materon_TF29_v3_6`
- 当前目录仅保留最终 v3.6 代码、验收标准、设计合规文件与验证报告。
- v3.1–v3.5 的分散 changelog 已合并进 provenance，不再作为独立文件保留。
- 科学模型名和协议字段未改名。


The engine package is intentionally isolated as `materon_tf29_engine` so it cannot shadow the project-level `materon` core. Engine scripts add their own directory to `sys.path`, so `bash scripts/preflight_all.sh 0` remains self-contained.

推荐入口：

```bash
bash scripts/preflight_all.sh 0
```

正式运行前必须先通过环境、官方 builder、权重配对、恒等性和 checkpoint 验收。
