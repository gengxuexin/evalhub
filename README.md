# EvalHub

多 Agent 评测平台。用同一套消防法规题，比较法规问答、统一入口，以及后续接入的其他系统。

它调用被测系统，保存当次回答和参考材料，再计算调用情况与回答质量。评测过程不会替被测系统重新检索。调用成功不代表回答正确。

## 运行

在 WSL 中：

```bash
cd /home/geng/fire_agent_v2/EvalHub
source /home/geng/.venv/bin/activate
python -m app.cli run --limit 3 --save-baseline fire_qa
python -m app.cli show results/eval_xxx.jsonl
python -m app.cli compare --baseline fire_qa --candidate results/eval_xxx.jsonl
python -m app.cli rerun-failed results/eval_xxx.jsonl
```

默认适配器是 `fire_agent`。超过 10 条样本需要 `--allow-large`。

对已有结果做 Judge 和向量评分，不重新调用 Agent：

```bash
python -m app.cli score results/eval_xxx.jsonl --judge --semantic
```

人工标注和本地网站：

```bash
python -m app.cli annotate <run_id> sample_0001 --label correct --comment "事实齐全"
python -m app.cli serve --port 8766
```

打开 `http://127.0.0.1:8766`。界面使用 [Tabler](https://github.com/tabler/tabler)（MIT）。页面包括总览、消防题库标准、Agent 接入方式、基准版本，以及单次运行的质量分和标注。标注追加写入 `results/annotations/<run_id>.jsonl`，同一样本以最后一次为准。网站只读取已保存的结果，不会在打开页面时调用 Agent。

保存这次运行的基准版本，并检查以后的运行是否还使用同一数据集、适配器、top_k 和 Judge 提示词：

```bash
python -m app.cli benchmark save --name fire_qa_v1 --run <run_id>
python -m app.cli benchmark check --name fire_qa_v1 --run <run_id>
python -m app.cli benchmark list
python -m app.cli agree results/eval_xxx.jsonl --sample sample_0001 --repeats 2
```

其他 HTTP JSON 服务可以在网站「接入 Agent」里登记，或直接指定字段名：

```bash
python -m app.cli agents
python -m app.cli run --agent fire_gateway --limit 3
python -m app.cli run --adapter http_json --url http://127.0.0.1:9000/ask \
  --query-field query --answer-field output --contexts-field docs --limit 1
```

`fire_agent` 是 Agent1 法规问答，`fire_gateway` 是网关并强制走法规问答。Agent2 语音和 Agent3 图像已列在目录里，当前文字题库不能直接评测。令牌只从环境变量读取。

## 一次运行留下的文件

- `results/eval_<run_id>.jsonl`：逐条结果，含回答、contexts、耗时、错误类别
- `results/reports/<run_id>/operational_report.json`：成功率与耗时
- `results/reports/<run_id>/quality_report.json`：字面质量，以及可选的 Judge 指标
- `results/reports/<run_id>/report.html`：可直接打开的结果页
- `results/reports/<run_id>/manifest.json`：数据集 SHA256、适配器、top_k
- `results/baselines/<name>.json`：基线指针，同名不会覆盖

## 指标边界

接口成功率不是答案正确率。精确匹配和字符 F1 只表示字面重合。余弦相似度没有通过阈值。切题分是三档映射。相关片段比例不考虑排序。上下文缺失、Judge 失败和拒评都不会被记成 0 分。

回归对比里，成功率下降或新出现的失败样本判定为回归。耗时和字面分波动只作提示。

## 已知限制

当前真实接入的是消防 Agent1。`http_json` 只支持一次 POST 返回 JSON。Judge 和向量相似度需要单独打开，默认运行不算这两项。完整 47 条数据集没有作为默认流程。
