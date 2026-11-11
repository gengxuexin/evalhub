"""评测网站：给使用方查看题库、评测对象、结果和人工标注。"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.annotations import LABELS, append_annotation, latest_annotations
from app.catalog import list_agents, probe_health, save_custom_agent
from app.view import _esc

STATIC_DIR = Path(__file__).resolve().parent / "static"
NAV = (
    ("/", "总览"),
    ("/standards", "评测题库"),
    ("/agents", "评测对象"),
    ("/benchmarks", "评测基准"),
)

METRIC_NAMES = {
    "normalized_exact_match": "字面是否一致",
    "character_f1": "字面重合",
    "semantic_similarity": "语义相近",
    "faithfulness": "是否依据材料",
    "answer_relevancy": "是否切题",
    "context_relevance_ratio": "材料相关比例",
    "context_average_precision": "材料排序质量",
    "context_recall": "材料是否覆盖要点",
}

LABEL_NAMES = {
    "correct": "正确",
    "partial": "部分正确",
    "incorrect": "不正确",
    "skip": "跳过",
}

STATUS_NAMES = {
    "scored": "已评分",
    "consistent": "一致",
    "inconsistent": "不一致",
    "not_comparable": "无法比较",
    "not_evaluable": "无法评分",
    "not_applicable": "不适用",
    "judge_failure": "评分失败",
    "judge_abstain": "拒绝评分",
    "invalid_output": "结果无效",
    "metric_failure": "评分出错",
}

AGENT_LABELS = {
    "fire_agent": "法规问答",
    "fire_gateway": "统一入口",
    "agent2_voice": "语音交互",
    "agent3_vision": "图像检查",
    "http_json": "其他服务",
}

COUNT_NAMES = {
    "ok": "成功",
    "auth_failed": "登录失败",
    "timeout": "超时",
    "connection_error": "连不上",
    "http_server_error": "服务出错",
    "http_client_error": "请求有误",
    "empty_answer": "没有回答",
    "agent_error": "评测对象报错",
    "unknown_failure": "未知问题",
}

DIFFICULTY_NAMES = {
    "easy": "容易",
    "medium": "中等",
    "hard": "较难",
}

METRIC_GUIDE = (
    ("字面是否一致", "和参考答案逐字相同才算满分。回答更完整、但比参考答案长，这项仍是零。"),
    ("字面重合", "看用了多少相同的字，不代表事实对不对。"),
    ("语义相近", "看意思是否接近，没有及格线。需要单独开启。"),
    ("是否依据材料", "只看这次评测对象带回的材料。没有材料就不评分。"),
    ("是否切题", "直接回答算满分，部分回答算一半，不相关算零。"),
    ("材料相关比例", "相关材料占全部材料的比例，不看先后顺序。"),
    ("材料排序质量", "相关材料越靠前越好。一条都不相关就是零。"),
    ("材料是否覆盖要点", "材料要盖住参考答案里的要点，只是话题相近不算覆盖。"),
)


def create_app(results_dir: Path) -> FastAPI:
    results_dir = Path(results_dir)
    annotations_dir = results_dir / "annotations"
    dataset_path = results_dir.parent / "data" / "datasets" / "fire_qa.jsonl"
    app = FastAPI(title="EvalHub")

    def runs() -> list[dict]:
        found = []
        report_root = results_dir / "reports"
        if not report_root.exists():
            return found
        for manifest_path in sorted(report_root.glob("*/manifest.json"), reverse=True):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("run_id"):
                found.append(manifest)
        return found

    def run_records(manifest: dict) -> list[dict]:
        path = Path(manifest["result_file"])
        rows = []
        if not path.is_file():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        manifests = runs()
        profile = _dataset_profile(dataset_path)
        known_agents = list_agents(results_dir / "agents")
        usable = sum(1 for item in known_agents if item.get("usable_with_fire_qa"))
        bench_count = len(list((results_dir / "benchmarks").glob("*.json"))) if (results_dir / "benchmarks").exists() else 0
        stats = "".join([
            _stat("题库题目", profile["total"] if profile["exists"] else "—", "消防法规问答", "purple"),
            _stat("评测对象", str(usable), "可以用这套题评测的数量", "blue"),
            _stat("已有评测", str(len(manifests)), "只展示已经保存的结果", "green"),
            _stat("评测基准", str(bench_count), "用来对照以后的评测", "red"),
        ])
        rows = []
        for manifest in manifests:
            run_id = str(manifest.get("run_id") or "")
            adapter = (manifest.get("adapter") or {}).get("name") or ""
            executed = (manifest.get("dataset") or {}).get("executed_samples")
            counts = manifest.get("failure_counts") or {}
            rows.append(
                "<tr>"
                f"<td><a href='/runs/{_esc(run_id)}'>{_esc(_when(manifest.get('created_at') or run_id))}</a></td>"
                f"<td>{_esc(_zh_agent(adapter))}</td>"
                f"<td>{_esc(executed)} 题</td>"
                f"<td>{_esc(_count_text(counts))}</td>"
                "</tr>"
            )
        table = (
            "<div class='table-responsive'><table class='table table-hover'>"
            "<thead><tr><th>评测时间</th><th>评测对象</th><th>题数</th><th>调用结果</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=4>还没有评测记录</td></tr>'}</tbody></table></div>"
        )
        side = (
            "<p>这里查看已经完成的评测。法规问答和统一入口可以用这套消防题评测。</p>"
            "<p>语音交互和图像检查也已接入，但它们要听录音或看画面，这套文字题先不评这两项。</p>"
            "<p><a class='btn btn-primary' href='/standards'>查看题库</a> "
            "<a class='btn btn-outline-primary' href='/agents'>查看评测对象</a></p>"
        )
        body = (
            f"<div class='row'>{stats}</div>"
            "<div class='row'>"
            f"<div class='col-lg-8'>{_card('评测记录', table)}</div>"
            f"<div class='col-lg-4'>{_card('说明', side)}</div>"
            "</div>"
        )
        return _layout("总览", "/", body)

    @app.get("/standards", response_class=HTMLResponse)
    def standards() -> str:
        profile = _dataset_profile(dataset_path)
        if not profile["exists"]:
            intro = "<p>还没有准备好消防题库。</p>"
        else:
            intro = (
                f"<p>共 {_esc(profile['total'])} 题，用来评测消防法规问答。"
                f"已核对 {_esc(profile['verified_true'])} 题，"
                f"尚未核对 {_esc(profile['verified_false'])} 题。</p>"
            )
        categories = _count_table(profile["categories"], "类别", "题数")
        difficulties = _count_table(profile["difficulties"], "难度", "题数", translate=_zh_difficulty)
        examples = []
        for item in profile["examples"]:
            examples.append(
                "<tr>"
                f"<td>{_esc(item.get('category'))}</td>"
                f"<td>{_esc(_zh_difficulty(item.get('difficulty')))}</td>"
                f"<td>{_esc(item.get('question'))}</td>"
                "</tr>"
            )
        example_table = (
            "<div class='table-responsive'><table class='table table-hover'>"
            "<thead><tr><th>类别</th><th>难度</th><th>问题</th></tr></thead>"
            f"<tbody>{''.join(examples)}</tbody></table></div>"
        )
        metrics = []
        for title, note in METRIC_GUIDE:
            metrics.append(
                "<tr>"
                f"<td>{_esc(title)}</td>"
                f"<td>{_esc(note)}</td>"
                "</tr>"
            )
        metric_table = (
            "<div class='table-responsive'><table class='table table-hover'>"
            "<thead><tr><th>评分项</th><th>怎么理解</th></tr></thead>"
            f"<tbody>{''.join(metrics)}</tbody></table></div>"
        )
        body = (
            "<div class='row'>"
            f"<div class='col-lg-8'>{_card('消防法规题库', intro + categories)}</div>"
            f"<div class='col-lg-4'>{_card('难度', difficulties)}</div>"
            f"<div class='col-12'>{_card('题目示例', example_table or '<p>题库为空。</p>')}</div>"
            f"<div class='col-12'>{_card('评分怎么看', metric_table)}"
            "<p class='text-muted mt-3'>调用成功只说明评测对象返回了回答。"
            "字面分和语义分都不是法律结论。没有材料、评分失败或拒绝评分，都不会被记成零分。</p></div>"
            "</div>"
        )
        return _layout("评测题库", "/standards", body)

    @app.get("/agents", response_class=HTMLResponse)
    def agents(error: str = "") -> str:
        cards = []
        for item in list_agents(results_dir / "agents"):
            status = probe_health(str(item.get("health_url") or ""))
            badge = "可以评测" if item.get("usable_with_fire_qa") else "这套题暂不评测"
            badge_class = "bg-success" if item.get("usable_with_fire_qa") else "bg-warning"
            body = (
                f"<p><span class='badge {badge_class}'>{badge}</span> "
                f"<span class='badge bg-light text-dark'>{_esc(item.get('kind'))}</span> "
                f"<span class='badge bg-primary'>{_esc(status)}</span></p>"
                f"<p>{_esc(item.get('summary'))}</p>"
            )
            cards.append(f"<div class='col-lg-6'>{_card(str(item.get('title') or '评测对象'), body)}</div>")
        notice = f"<div class='alert alert-danger'>{_esc(error)}</div>" if error else ""
        form = (
            "<form method='post' action='/agents' class='row g-3'>"
            "<div class='col-md-4'><label class='form-label'>名称</label>"
            "<input class='form-control' name='name' placeholder='给它起个名字' required></div>"
            "<div class='col-md-8'><label class='form-label'>服务地址</label>"
            "<input class='form-control' name='url' placeholder='填写它接收问题的地址' required></div>"
            "<div class='col-md-3'><label class='form-label'>问题放在</label>"
            "<input class='form-control' name='query_field' placeholder='留空则使用默认位置'></div>"
            "<div class='col-md-3'><label class='form-label'>回答放在</label>"
            "<input class='form-control' name='answer_field' placeholder='留空则使用默认位置'></div>"
            "<div class='col-md-3'><label class='form-label'>材料放在</label>"
            "<input class='form-control' name='contexts_field' placeholder='留空则使用默认位置'></div>"
            "<div class='col-md-3'><label class='form-label'>成功标记</label>"
            "<input class='form-control' name='success_value' placeholder='留空则按默认标记'></div>"
            "<div class='col-md-4'><label class='form-label'>口令变量名</label>"
            "<input class='form-control' name='token_env' placeholder='只写变量名，不要写口令'></div>"
            "<div class='col-md-4'><label class='form-label'>口令放在</label>"
            "<input class='form-control' name='token_header' placeholder='对方要求的请求头'></div>"
            "<div class='col-md-4'><label class='form-label'>口令方式</label>"
            "<select class='form-select' name='token_prefix'>"
            "<option value=''>原样发送</option><option value='Bearer'>按登录令牌发送</option></select></div>"
            "<div class='col-12'><button class='btn btn-primary' type='submit'>保存评测对象</button></div>"
            "</form>"
            "<p class='text-muted mt-3 mb-0'>只保存地址和字段位置。口令留在环境变量里，页面不会保存口令。</p>"
        )
        body = (
            notice
            + "<div class='row'>"
            + "".join(cards)
            + f"<div class='col-12'>{_card('登记其他评测对象', form)}</div></div>"
        )
        return _layout("评测对象", "/agents", body)

    @app.post("/agents")
    def add_agent(
        name: str = Form(...),
        url: str = Form(...),
        query_field: str = Form("question"),
        answer_field: str = Form("answer"),
        contexts_field: str = Form("contexts"),
        success_value: str = Form("success"),
        token_env: str = Form(""),
        token_header: str = Form(""),
        token_prefix: str = Form(""),
    ) -> RedirectResponse:
        try:
            save_custom_agent(results_dir / "agents", {
                "name": name,
                "url": url,
                "query_field": query_field,
                "answer_field": answer_field,
                "contexts_field": contexts_field,
                "success_value": success_value,
                "token_env": token_env,
                "token_header": token_header,
                "token_prefix": token_prefix,
            })
        except ValueError as exc:
            return RedirectResponse(f"/agents?error={quote(str(exc))}", status_code=303)
        return RedirectResponse("/agents", status_code=303)

    @app.get("/benchmarks", response_class=HTMLResponse)
    def benchmarks() -> str:
        body = (
            "<div class='row'>"
            f"<div class='col-12'>{_card('评测基准', _benchmark_table(results_dir / 'benchmarks'))}</div>"
            f"<div class='col-12'>{_card('结果对照', _baseline_table(results_dir / 'baselines'))}</div>"
            "</div>"
        )
        return _layout("评测基准", "/benchmarks", body)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(run_id: str) -> str:
        manifest = _find_run(runs(), run_id)
        if manifest is None:
            return _layout("未找到", "/", "<div class='card'><div class='card-body'><p>没有这次评测。</p></div></div>")
        labels = latest_annotations(annotations_dir, run_id)
        report_dir = results_dir / "reports" / run_id
        quality = _load_json(report_dir / "quality_judge_report.json")
        if not quality:
            quality = _load_json(report_dir / "quality_report.json")
        summary = _summary_table(quality.get("summary") or {})
        versions = quality.get("versions") or {}
        version_line = "<p class='text-muted'>这次使用的评分规则已记录。</p>" if versions else ""
        adapter = manifest.get("adapter") or {}
        dataset = manifest.get("dataset") or {}
        meta = (
            f"<p>评测对象：{_esc(_zh_agent(adapter.get('name')))}</p>"
            f"<p>完成 {_esc(dataset.get('executed_samples'))} 题，题库共 {_esc(dataset.get('total_samples'))} 题。</p>"
            f"<p>调用结果：{_esc(_count_text(manifest.get('failure_counts') or {}))}</p>"
        )
        operational = _operational_card(manifest.get("operational_report"))
        blocks = [
            "<div class='row mb-3'>",
            f"<div class='col-lg-7'>{_card('这次评测', meta + version_line)}</div>",
            f"<div class='col-lg-5'>{operational}</div>",
            "</div>",
            _card("质量汇总", summary),
            _agreement_table(report_dir, run_id),
        ]
        by_sample = {
            item.get("sample_id"): item.get("metrics") or {}
            for item in quality.get("samples") or []
        }
        sample_cards = []
        for record in run_records(manifest):
            sample_id = str(record.get("sample_id") or "")
            current = labels.get(sample_id) or {}
            options = []
            for label in LABELS:
                selected = " selected" if current.get("label") == label else ""
                options.append(
                    f"<option value='{_esc(label)}'{selected}>{_esc(_zh_label(label))}</option>"
                )
            answer = str(record.get("answer") or "")[:500]
            scores = _score_line(by_sample.get(sample_id) or {})
            comment = current.get("comment") or ""
            success = "成功" if record.get("success") is True else "失败"
            shown_label = _zh_label(current.get("label")) if current.get("label") else "未标注"
            sample_cards.append(
                "<div class='card'>"
                "<div class='card-header d-flex justify-content-between align-items-center'>"
                f"<h4 class='mb-0'>{_esc(_zh_sample(sample_id))}</h4>"
                f"<span class='badge bg-secondary'>{success}</span></div>"
                "<div class='card-body'>"
                f"<p class='text-muted'>参考材料 {len(record.get('contexts') or [])} 条</p>"
                f"<p><strong>问题</strong> {_esc(record.get('question'))}</p>"
                f"<p><strong>参考答案</strong> {_esc(record.get('ground_truth'))}</p>"
                f"<p><strong>回答</strong></p><div class='answer-text'>{_esc(answer)}</div>"
                f"<p class='score-line mt-3'>评分：{_esc(scores)}</p>"
                f"<p>当前标注：{_esc(shown_label)}"
                f"{'，' + _esc(comment) if comment else ''}</p>"
                "<form method='post' action='/annotations' class='row g-2'>"
                f"<input type='hidden' name='run_id' value='{_esc(run_id)}'>"
                f"<input type='hidden' name='sample_id' value='{_esc(sample_id)}'>"
                f"<div class='col-md-3'><select class='form-select' name='label'>{''.join(options)}</select></div>"
                "<div class='col-md-6'><input class='form-control' name='comment' placeholder='备注'></div>"
                "<div class='col-md-3'><button class='btn btn-primary w-100' type='submit'>保存标注</button></div>"
                "</form></div></div>"
            )
        blocks.append(f"<div class='row mt-3'><div class='col-12'>{''.join(sample_cards)}</div></div>")
        title = f"{_zh_agent(adapter.get('name'))} · {_when(manifest.get('created_at') or run_id)}"
        return _layout(title, "/", "".join(blocks))

    @app.post("/annotations")
    def save_annotation(
        run_id: str = Form(...),
        sample_id: str = Form(...),
        label: str = Form(...),
        comment: str = Form(""),
    ) -> RedirectResponse:
        manifest = _find_run(runs(), run_id)
        if manifest is None:
            return RedirectResponse("/", status_code=303)
        known = {str(item.get("sample_id")) for item in run_records(manifest)}
        if sample_id not in known:
            return RedirectResponse(f"/runs/{run_id}", status_code=303)
        append_annotation(
            annotations_dir,
            run_id=run_id,
            sample_id=sample_id,
            label=label,
            comment=comment,
        )
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _dataset_profile(path: Path) -> dict:
    profile = {
        "path": str(path),
        "exists": False,
        "total": 0,
        "verified_true": 0,
        "verified_false": 0,
        "categories": {},
        "difficulties": {},
        "examples": [],
    }
    if not path.is_file():
        return profile
    profile["exists"] = True
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        profile["total"] += 1
        category = str(row.get("category") or "未分类")
        difficulty = str(row.get("difficulty") or "未标")
        profile["categories"][category] = profile["categories"].get(category, 0) + 1
        profile["difficulties"][difficulty] = profile["difficulties"].get(difficulty, 0) + 1
        if row.get("verified") is True:
            profile["verified_true"] += 1
        else:
            profile["verified_false"] += 1
        if len(profile["examples"]) < 5:
            profile["examples"].append({
                "question": row.get("question"),
                "category": category,
                "difficulty": difficulty,
            })
    return profile


def _count_table(counts: dict, left: str, right: str, translate=None) -> str:
    rows = []
    for name, count in counts.items():
        shown = translate(name) if translate else name
        rows.append(f"<tr><td>{_esc(shown)}</td><td>{_esc(count)}</td></tr>")
    if not rows:
        return "<p>没有可统计的题目。</p>"
    return (
        "<table class='table table-vcenter card-table'>"
        f"<thead><tr><th>{_esc(left)}</th><th>{_esc(right)}</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _benchmark_table(directory: Path) -> str:
    if not directory.is_dir():
        return "<p>还没有评测基准。</p>"
    rows = []
    for path in sorted(directory.glob("*.json")):
        item = _load_json(path)
        if not item:
            continue
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('name'))}</td>"
            f"<td><a href='/runs/{_esc(item.get('source_run_id'))}'>{_esc(_when(item.get('created_at') or item.get('source_run_id')))}</a></td>"
            f"<td>{_esc(_zh_agent(item.get('adapter_name')))}</td>"
            f"<td>评分规则已记录</td>"
            "</tr>"
        )
    if not rows:
        return "<p>还没有评测基准。</p>"
    return (
        "<div class='table-responsive'><table class='table table-hover'>"
        "<thead><tr><th>名称</th><th>来源评测</th><th>评测对象</th><th>评分规则</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "<p class='text-muted mb-0'>基准用来确认以后的评测还是同一套题、同一个评测对象和同一套评分规则。对不上只说明不能直接比较，不表示回答变差。</p>"
    )


def _baseline_table(directory: Path) -> str:
    if not directory.is_dir():
        return "<p>还没有结果对照。</p>"
    rows = []
    for path in sorted(directory.glob("*.json")):
        item = _load_json(path)
        if not item:
            continue
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('name'))}</td>"
            f"<td><a href='/runs/{_esc(item.get('run_id'))}'>{_esc(_when(item.get('created_at') or item.get('run_id')))}</a></td>"
            f"<td>{_esc(_zh_agent((item.get('adapter') or {}).get('name')))}</td>"
            "</tr>"
        )
    if not rows:
        return "<p>还没有结果对照。</p>"
    return (
        "<table class='table table-hover'>"
        "<thead><tr><th>名称</th><th>评测</th><th>评测对象</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _operational_card(path_value) -> str:
    report = _load_json(Path(path_value)) if path_value else {}
    metrics = report.get("metrics") or {}
    if not metrics:
        return _card("调用情况", "<p>还没有调用记录。</p>")
    rate = metrics.get("success_rate")
    shown = "-" if rate is None else f"{float(rate) * 100:.0f}%"
    latency = metrics.get("average_latency_ms")
    latency_text = "-" if latency is None else f"{float(latency):.0f} 毫秒"
    body = (
        f"<div class='h1 mb-1'>{_esc(shown)}</div>"
        "<div class='text-secondary mb-2'>调用成功率</div>"
        f"<div>平均耗时 {_esc(latency_text)}</div>"
        f"<div class='text-secondary'>成功 {_esc(metrics.get('success_count'))}，"
        f"失败 {_esc(metrics.get('failure_count'))}</div>"
    )
    return _card("调用情况", body)


def _load_json(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _agreement_table(report_dir: Path, run_id: str) -> str:
    paths = sorted(report_dir.glob("agreement_*.json"))
    if not paths:
        return ""
    rows = []
    for path in paths:
        report = _load_json(path)
        sample_id = report.get("sample_id") or path.stem
        for name, item in (report.get("metrics") or {}).items():
            if name in {"normalized_exact_match", "character_f1"}:
                continue
            rows.append(
                "<tr>"
                f"<td>{_esc(_zh_sample(sample_id))}</td>"
                f"<td>{_esc(_zh_metric(name))}</td>"
                f"<td>{_esc(_zh_status(item.get('status')))}</td>"
                f"<td>{_esc(item.get('min_score'))}</td>"
                f"<td>{_esc(item.get('max_score'))}</td>"
                "</tr>"
            )
    if not rows:
        return ""
    table = (
        "<div class='table-responsive'><table class='table table-vcenter card-table'>"
        "<thead><tr><th>题目</th><th>评分项</th><th>是否一致</th><th>最低</th><th>最高</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    return _card("重复评分", table)


def _summary_table(summary: dict) -> str:
    if not summary:
        return "<p>还没有质量评分。</p>"
    rows = []
    for name, item in summary.items():
        mean = item.get("mean_score")
        shown = "-" if mean is None else f"{float(mean):.3f}"
        rows.append(
            "<tr>"
            f"<td>{_esc(_zh_metric(name))}</td>"
            f"<td>{_esc(item.get('scored_count'))}</td>"
            f"<td>{_esc(shown)}</td>"
            "</tr>"
        )
    return (
        "<div class='table-responsive'><table class='table table-vcenter card-table'>"
        "<thead><tr><th>指标</th><th>已评分</th><th>平均分</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _score_line(metrics: dict) -> str:
    if not metrics:
        return "无"
    parts = []
    for name, item in metrics.items():
        score = item.get("score")
        shown = "-" if score is None else f"{float(score):.3f}"
        parts.append(f"{_zh_metric(name)} {shown}")
    return "；".join(parts)


def _count_text(counts: dict) -> str:
    if not counts:
        return "-"
    return "，".join(f"{COUNT_NAMES.get(str(key), '其他')} {value}" for key, value in counts.items())


def _find_run(manifests: list[dict], run_id: str) -> dict | None:
    for manifest in manifests:
        if manifest.get("run_id") == run_id:
            return manifest
    return None


def _card(title: str, body: str) -> str:
    return (
        "<div class='card'><div class='card-header'>"
        f"<h3 class='card-title'>{_esc(title)}</h3></div>"
        f"<div class='card-body'>{body}</div></div>"
    )


def _stat(label: str, value: str, hint: str, color: str) -> str:
    return (
        f"<div class='col-6 col-lg-3'><div class='card'><div class='card-body px-3 py-4'><div class='row'>"
        f"<div class='col-4 d-flex align-items-center justify-content-center'>"
        f"<div class='stats-icon {color}'></div></div>"
        "<div class='col-8'>"
        f"<h6 class='text-muted font-semibold'>{_esc(label)}</h6>"
        f"<h4 class='font-extrabold mb-0'>{_esc(value)}</h4>"
        f"<p class='mb-0'>{_esc(hint)}</p>"
        "</div></div></div></div></div>"
    )


def _zh_metric(name: str) -> str:
    return METRIC_NAMES.get(str(name), "其他评分")


def _zh_label(name: str) -> str:
    return LABEL_NAMES.get(str(name), "未标注")


def _zh_status(name) -> str:
    return STATUS_NAMES.get(str(name), "已记录")


def _zh_agent(name) -> str:
    text = str(name or "")
    return AGENT_LABELS.get(text, text or "其他评测对象")


def _zh_sample(sample_id) -> str:
    text = str(sample_id or "")
    if text.startswith("sample_") and text[7:].isdigit():
        return f"第{int(text[7:])}题"
    return "题目"


def _zh_difficulty(value) -> str:
    return DIFFICULTY_NAMES.get(str(value or ""), "未标")


def _when(value) -> str:
    text = str(value or "")
    parsed = None
    if "T" in text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    elif len(text) >= 15 and text[:8].isdigit() and text[8] == "_":
        try:
            parsed = datetime.strptime(text[:15], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            parsed = None
    if parsed is None:
        return "时间未记录"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone(timezone(timedelta(hours=8)))
    return f"{local.year}年{local.month}月{local.day}日 {local.hour:02d}:{local.minute:02d}"


def _layout(title: str, active: str, body: str) -> str:
    links = []
    for href, label in NAV:
        current = " active" if href == active else ""
        links.append(
            f"<li class='sidebar-item{current}'><a class='sidebar-link' href='{href}'>"
            f"<span>{_esc(label)}</span></a></li>"
        )
    return (
        "<!DOCTYPE html><html lang='zh-CN'><head>"
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_esc(title)} · 消防评测</title>"
        "<link rel='stylesheet' href='/static/vendor/css/app.css'>"
        "<link rel='stylesheet' href='/static/site.css'>"
        "</head><body><div id='app'>"
        "<div id='sidebar' class='active'><div class='sidebar-wrapper active'>"
        "<div class='sidebar-header'><div class='d-flex justify-content-between'>"
        "<div class='logo'><a href='/'><span>消防评测</span></a></div>"
        "</div></div>"
        "<div class='sidebar-menu'><ul class='menu'>"
        "<li class='sidebar-title'>评测</li>"
        + "".join(links)
        + "</ul></div></div></div>"
        "<div id='main'><div class='page-heading'>"
        f"<h3>{_esc(title)}</h3>"
        "<p class='text-subtitle text-muted'>查看消防评测对象的回答效果</p></div>"
        f"<div class='page-content'>{body}</div>"
        "<footer><div class='footer clearfix mb-0 text-muted'>"
        "<p>页面按使用方查看结果来组织。评测保存的是评测对象当次返回的回答和参考材料。</p>"
        "</div></footer></div></div>"
        "<script src='/static/vendor/js/app.js'></script></body></html>"
    )
