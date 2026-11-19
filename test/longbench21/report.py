"""Offline paired reporting for the frozen LongBench v1 benchmark (standard library only)."""
from collections import Counter, defaultdict
from html import escape
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile


# Function names from the official LongBench v1 dataset2metric registry.
METRICS = {
    "hotpotqa": "qa_f1_score", "2wikimqa": "qa_f1_score", "musique": "qa_f1_score",
    "dureader": "rouge_zh_score", "narrativeqa": "qa_f1_score", "qasper": "qa_f1_score",
    "multifieldqa_en": "qa_f1_score", "multifieldqa_zh": "qa_f1_zh_score",
    "gov_report": "rouge_score", "qmsum": "rouge_score", "multi_news": "rouge_score",
    "vcsum": "rouge_zh_score", "trec": "classification_score", "triviaqa": "qa_f1_score",
    "samsum": "rouge_score", "lsht": "classification_score", "passage_count": "count_score",
    "passage_retrieval_en": "retrieval_score", "passage_retrieval_zh": "retrieval_zh_score",
    "lcc": "code_sim_score", "repobench-p": "code_sim_score",
}
METRIC_LABELS = {
    "qa_f1_score": "English QA F1", "qa_f1_zh_score": "Chinese QA F1",
    "rouge_score": "English ROUGE-L", "rouge_zh_score": "Chinese ROUGE-L",
    "code_sim_score": "Code similarity", "classification_score": "Classification score",
    "retrieval_score": "English retrieval score", "retrieval_zh_score": "Chinese retrieval score",
    "count_score": "Counting score",
}
# These official task templates supply a shared instruction instead of a per-example question.
TASK_INSTRUCTIONS = {
    "gov_report": "Write a one-page summary of the report.",
    "multi_news": "Now, write a one-page summary of all the news.",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。",
    "passage_count": "Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.",
    "lcc": "Please complete the code given below.",
}
# All 100 frozen tasks in each classification subset use the same ordered list.
CLASS_LABELS = {'trec': ('Food', 'Date', 'Order, rank', 'Speed', 'Disease and medicine',
          'Word with a special property', 'Abbreviation', 'Language', 'Letter like a-z',
          'Other entity', 'Animal', 'Expression abbreviated', 'Price', 'Techniques and method',
          'Musical instrument', 'Mountain', 'Currency name', 'Event', 'Product', 'State',
          'Individual', 'Organ of body', 'Reason', 'Manner of an action', 'City', 'Religion',
          'Invention, book and other creative piece', 'Distance, linear measure', 'Temperature',
          'Postcode or other code', 'Size, area and volume', 'Sport', 'Country', 'Other location',
          'Lasting time of somethin', 'Equivalent term', 'Description of something', 'Weight',
          'Vehicle', 'Color', 'Other number', 'Definition of something', 'Element and substance',
          'Description of a person', 'Symbols and sign', 'Number of something', 'Plant',
          'Percent, fraction', 'Group or organization of person', 'Title of a person'),
 'lsht': ('农业、农村', '军事', '文学、艺术', '体育', '传媒业', '电子信息产业', '文化、休闲娱乐', '社会、劳动', '经济', '服务业、旅游业', '环境、气象',
          '能源、水务、水利', '财政、金融', '教育', '科学技术', '对外关系、国际关系', '矿业、工业', '政治', '交通运输、邮政、物流', '灾难、事故',
          '基本建设、建筑业、房地产', '医药、卫生', '法律、司法', '商业、外贸、海关')}
METHODS = ("Full", "RedKnot")
BANDS = ("16k", "32k", "64k", "128k", "256k")
MODES = ("native", "extended")
DEFAULT_COUNTS = dict.fromkeys(METRICS, 100)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def expectations(counts):
    expected = dict(DEFAULT_COUNTS if counts is None else counts)
    for dataset, n in expected.items():
        require(dataset in METRICS and type(n) is int and n >= 0, f"Invalid target count for {dataset}")
    return expected


def public_methods(values):
    require(isinstance(values, dict), "Expected a mapping of methods")
    keys = set(values)
    require(keys in ({"Full","RedKnot"}, {"Full","S1.6"}), f"Expected exactly two methods; found {sorted(keys)}")
    candidate = "RedKnot" if "RedKnot" in values else "S1.6"
    return {"Full": values["Full"], "RedKnot": values[candidate]}


def validate_rows(rows, *, expected_counts=None, score_fn=None):
    """Validate committed target-only group_scores rows and return independent, public-name rows.

    score_fn(row, prediction_text) optionally recomputes the official per-example metric on [0,1].
    Raw rows and a prior answers.jsonl have the same field names. Only the candidate arm key changes.
    """
    expected = expectations(expected_counts)
    seen_ids, seen_sources = set(), set()
    counts = Counter()
    normalized = []
    for raw in rows:
        require(isinstance(raw, dict), "Expected a scored row object")
        row = dict(raw)
        dataset, index = row.get("dataset"), row.get("index")
        require(dataset in expected and type(index) is int and index >= 0, f"Unexpected target {dataset}/{index}")
        name = f"{dataset}/{index}"
        sid = row.get("source_id")
        require(isinstance(sid, str) and sid and type(row.get("source_index")) is int, f"Missing source identity in {name}")
        require(index not in seen_ids and (dataset, sid) not in seen_sources, f"Duplicate target {name}")
        seen_ids.add(index)
        seen_sources.add((dataset, sid))
        counts[dataset] += 1
        require(counts[dataset] <= expected[dataset], f"Excess targets in {dataset}")
        require(row.get("metric") == METRICS[dataset], f"Wrong official metric in {name}")
        require(isinstance(row.get("question"), str) and isinstance(row.get("references"), list)
                and row["references"] and all(isinstance(v, str) for v in row["references"]), f"Missing QA fields in {name}")
        if dataset in CLASS_LABELS:
            classes = list(CLASS_LABELS[dataset])
            if "all_classes" in row:
                require(row["all_classes"] == classes, f"Unexpected classification vocabulary in {name}")
            require(set(row["references"]).issubset(classes), f"Unknown reference class in {name}")
            row["all_classes"] = classes
        if not row["question"] and dataset in TASK_INSTRUCTIONS:
            row["task_instruction"] = TASK_INSTRUCTIONS[dataset]
        require(row.get("length_band") in BANDS and row.get("composition_mode") in MODES, f"Unknown input group in {name}")
        require(type(row.get("input_tokens")) is int and row["input_tokens"] > 0 and is_hash(row.get("input_ids_sha256"))
                and is_hash(row.get("manifest_file_sha256")), f"Missing input binding in {name}")
        require(type(row.get("chunk_count")) is int and 5 <= row["chunk_count"] <= 15
                and isinstance(row.get("group"), str) and row["group"] and row.get("batch_size") in (4,8)
                and type(row.get("owner_rank")) is int and row["owner_rank"] >= 0, f"Missing batch details in {name}")
        row["arms"] = {m:dict(a) for m,a in public_methods(row.get("arms")).items()}
        if "native_files" in row:
            row["native_files"] = public_methods(row["native_files"])
            require(all(is_hash(x) for x in row["native_files"].values()), f"Invalid raw file hashes in {name}")
        for m,a in row["arms"].items():
            require(isinstance(a.get("prediction_text"), str) and isinstance(a.get("complete_raw_text"), str)
                    and type(a.get("output_tokens")) is int and a["output_tokens"] > 0
                    and is_hash(a.get("output_ids_sha256")) and isinstance(a.get("finish_reason"),str) and a["finish_reason"],
                    f"Incomplete output in {name}/{m}")
            require(is_number(a.get("ttft_s")) and is_number(a.get("duration_s")) and 0 < a["ttft_s"] <= a["duration_s"],
                    f"Invalid timing in {name}/{m}")
            require(is_number(a.get("score")) and 0 <= a["score"] <= 1, f"Invalid score in {name}/{m}")
            if score_fn is not None:
                recomputed = score_fn(row, a["prediction_text"])
                require(math.isclose(recomputed, a["score"],abs_tol=1e-10,rel_tol=1e-10),f"Official score mismatch in {name}/{m}")
        f, r = (row["arms"][m] for m in METHODS)
        delta = 100 * (r["score"] - f["score"])
        speedup = f["ttft_s"] / r["ttft_s"]
        for key,actual in (("score_change_pp",delta),("speedup_vs_full",speedup)):
            if key in row:
                require(is_number(row[key]) and math.isclose(row[key],actual,abs_tol=1e-8,rel_tol=1e-9), f"Incorrect {key} in {name}")
            row[key] = actual
        divergence = row.get("first_divergence_token_1based")
        require("first_divergence_token_1based" in row and
                (divergence is None or type(divergence) is int and 1 <= divergence <= min(f["output_tokens"],r["output_tokens"])+1)
                and (divergence is None)==(f["output_ids_sha256"]==r["output_ids_sha256"]), f"Invalid divergence in {name}")
        normalized.append(row)
    return sorted(normalized, key=lambda x:x["index"])


def load_jsonl(path, *, expected_counts=None, score_fn=None):
    """Load a previous public log; an empty file represents zero completed pairs."""
    path=Path(path)
    with path.open(encoding="utf-8") as stream:
        def records():
            for number,line in enumerate(stream,1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSON at {path}:{number}") from error
        return validate_rows(records(),expected_counts=expected_counts,score_fn=score_fn)


def aggregate(rows,metric=None):
    n=len(rows)
    counts=[r["input_tokens"] for r in rows]
    summary={"paired":n,"metric":metric,"min_input_tokens":min(counts,default=None),
             "max_input_tokens":max(counts,default=None),"mean_input_tokens":mean(counts) if n else None,
             "identical_outputs":sum(r["first_divergence_token_1based"] is None for r in rows),
             "higher":sum(r["score_change_pp"]>0 for r in rows),
             "equal":sum(r["score_change_pp"]==0 for r in rows),
             "lower":sum(r["score_change_pp"]<0 for r in rows)}
    for m in METHODS:
        arm=[r["arms"][m] for r in rows]
        out={"mean_ttft_s":mean(v["ttft_s"] for v in arm) if n else None,
             "mean_duration_s":mean(v["duration_s"] for v in arm) if n else None,
             "mean_output_tokens":mean(v["output_tokens"] for v in arm) if n else None,
             "termination":dict(Counter(v["finish_reason"] for v in arm))}
        if metric:
            out["score_100"]=100*mean(v["score"] for v in arm) if n else None
        summary[m]=out
    summary["speedup_ratio_of_mean_ttft"]=summary["Full"]["mean_ttft_s"]/summary["RedKnot"]["mean_ttft_s"] if n else None
    if metric:
        summary["score_change_pp"]=mean(r["score_change_pp"] for r in rows) if n else None
    return summary


def build_summary(rows,expected,mode):
    by_dataset=defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    planned_metrics={METRICS[k] for k,n in expected.items() if n}
    overall_metric=next(iter(planned_metrics)) if len(planned_metrics)==1 else None
    def groups(members,key,labels,metric):
        return {value:aggregate([r for r in members if r[key]==value],metric) for value in labels}
    datasets={k:aggregate(by_dataset[k],METRICS[k]) for k in expected}
    for dataset, stats in datasets.items():
        lower = (r for r in by_dataset[dataset] if r["score_change_pp"] < 0)
        largest = sorted(lower, key=lambda r: (r["score_change_pp"], r["index"]))[:2]
        stats["largest_score_drops"] = [
            {"index": r["index"], "score_change_pp": r["score_change_pp"]} for r in largest
        ]
    return {"status":"complete" if all(v["paired"]==expected[k] for k,v in datasets.items()) else "partial",
            "mode":mode,"expected":sum(expected.values()),"expected_by_dataset":expected,"completed":len(rows),
            "overall":aggregate(rows,overall_metric),"by_dataset":datasets,
            "by_length":groups(rows,"length_band",BANDS,overall_metric),"by_composition":groups(rows,"composition_mode",MODES,overall_metric),
            "by_dataset_and_length":{k:groups(by_dataset[k],"length_band",BANDS,METRICS[k]) for k in expected},
            "by_dataset_and_composition":{k:groups(by_dataset[k],"composition_mode",MODES,METRICS[k]) for k in expected}}


def write_atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,mode="w",encoding="utf-8",prefix="."+path.name,delete=False) as f:
            tmp=Path(f.name)
            f.write(payload)
        os.replace(tmp,path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def format_num(value,digits=2,suffix=""):
    return "—" if value is None else f"{value:,.{digits}f}{suffix}"


def table(groups):
    columns=("组别","配对题数","实际输入范围（token）","平均输入 token","官方指标","Full 得分","RedKnot 得分","分差","Full TTFT（秒）","RedKnot TTFT（秒）","TTFT 加速")
    lines=['<div class="table"><table><thead><tr>',"".join("<th>"+escape(h)+"</th>" for h in columns),"</tr></thead><tbody>"]
    for name,s in groups:
        f,r=(s[m] for m in METHODS)
        rng=f'{s["min_input_tokens"]:,}–{s["max_input_tokens"]:,}' if s["paired"] else "—"
        cells=[name,str(s["paired"]),rng,format_num(s["mean_input_tokens"],0),METRIC_LABELS.get(s["metric"],"—"),
               format_num(f.get("score_100")),format_num(r.get("score_100")),format_num(s.get("score_change_pp")),
               format_num(f["mean_ttft_s"],3),format_num(r["mean_ttft_s"],3),format_num(s["speedup_ratio_of_mean_ttft"],2,"×")]
        lines.append("<tr>"+"".join("<td>"+escape(c)+"</td>" for c in cells)+"</tr>")
    return "".join(lines)+"</tbody></table></div>"


STYLE="""*{box-sizing:border-box}body{margin:0;background:#f6f8f7;color:#173039;font:16px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1320px;margin:auto;padding:32px clamp(15px,3.4vw,50px) 70px}
h1{font-size:clamp(30px,4vw,46px);line-height:1.2;margin:0 0 8px}h2{margin:30px 0 10px;font-size:24px}h3{font-size:17px}a{color:#00695a;text-underline-offset:3px}nav{margin:12px 0}.note{font-size:14px;color:#536469}.table{overflow:auto;max-width:100%;background:white;border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:9px;text-align:right;border-bottom:1px solid #d7e3e2;white-space:nowrap}th{background:#eaf1ef}td:first-child,th:first-child{text-align:left}
details{background:white;border:1px solid #cbdbd8;border-radius:8px;margin:8px 0}summary{padding:12px;cursor:pointer}.detail{padding:0 14px 15px}.text{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}
.answers{display:grid;gap:12px;grid-template-columns:repeat(2,minmax(0,1fr))}.answers article{padding:12px;background:#f2f6f5;border-radius:6px}.answers h3{margin:0}.detail p{margin:7px 0 12px}
@media(max-width:760px){.answers{grid-template-columns:1fr}main{padding:18px 12px}}"""


def case(row):
    i=row["index"]
    q=row["question"] or row.get("task_instruction","")
    short=" ".join(q.split())
    if len(short)>92:
        short=short[:91]+"…"
    mode_labels={"native":"原生提示","extended":"扩展拼接"}
    bits=[f'<details id="q{i}"><summary><b>#{i}　{escape(short)}</b>　<span class="note">{escape(row["dataset"])} · '
          f'{escape(row["length_band"].upper())} · Δ {row["score_change_pp"]:+.2f} 分 · {row["speedup_vs_full"]:.2f}×</span></summary><div class="detail">',
          f'<p class="note">来源 ID {escape(row["source_id"])} · 数据集内行号 {row["source_index"]} · {row["input_tokens"]:,} 个输入 token · '
          f'{row["chunk_count"]} 个自然块 · {mode_labels[row["composition_mode"]]} · {escape(row["group"])}（B{row["batch_size"]}）</p>',
          "<h3>"+("原问题" if row["question"] else "任务指令")+"</h3>",
          '<p class="text">'+escape(q)+'</p><h3>参考答案</h3><ol>',
          "".join('<li class="text">'+escape(v)+"</li>" for v in row["references"]),'</ol><div class="answers">']
    for m in METHODS:
        a=row["arms"][m]
        bits.append(f'<article><h3>{m}</h3><p class="note">{escape(METRIC_LABELS[row["metric"]])} {a["score"]*100:.2f} 分 · '
                    f'TTFT {a["ttft_s"]:.3f} 秒 · 总耗时 {a["duration_s"]:.3f} 秒 · {a["output_tokens"]} 个输出 token · 结束原因 {escape(a["finish_reason"])}</p>'
                    f'<pre class="text">{escape(a["prediction_text"])}</pre></article>')
    return "".join(bits)+"</div></div></details>"


def html_page(rows,s):
    state="已完成" if s["status"]=="complete" else "部分结果"
    title="LongBench · Full 与 RedKnot"
    mode_labels={"inference":"本次推理","reference":"历史结果重放","native":"原生提示","extended":"扩展拼接"}
    mode=mode_labels.get(s["mode"],s["mode"])
    all_scores=s["overall"]
    output=[f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>{STYLE}</style></head><body><main><h1>{title}</h1>'
            f'<p>{state} · {s["completed"]:,} / {s["expected"]:,} 道配对题 · {escape(mode)}</p>',
            '<nav><a href="#datasets">按数据集</a>　<a href="#length">按输入长度</a>　<a href="#answers">逐题完整回答</a>　'
            '<a href="summary.json">汇总 JSON</a>　<a href="answers.jsonl">完整记录 JSONL</a></nav>',
            '<h2 id="datasets">各数据集的精度和首 token 延迟</h2>',table([("合计",s["overall"]),*s["by_dataset"].items()]),
            f'<p>按每题的官方分数比较，RedKnot 较高 {all_scores["higher"]} 题，同分 {all_scores["equal"]} 题，较低 {all_scores["lower"]} 题。</p>',
            '<p class="note">得分为各数据集官方指标的算术均值，按百分制展示。分差为 RedKnot 减 Full。逐题计数按舍入前的分数比较。采用不同指标的组合仅汇总可比的计数和耗时。'
            'TTFT 从收到请求计至首个 token，离线 KV 构建另计。加速倍数等于同一组 Full 平均 TTFT 除以 RedKnot 平均 TTFT。</p>',
            '<h2 id="length">按输入长度和提示构成</h2>',
            table([(k.upper(),v) for k,v in s["by_length"].items()]+[(mode_labels[k],v) for k,v in s["by_composition"].items()]),
            '<h2>各数据集内的长度与提示构成</h2>']
    for d,parts in s["by_dataset_and_length"].items():
        stats=s["by_dataset"][d]
        if stats["paired"]:
            output.append(f'<details><summary>{escape(d)} · {stats["paired"]} 道配对题 · RedKnot 较高 {stats["higher"]} 题，同分 {stats["equal"]} 题，较低 {stats["lower"]} 题</summary>')
            if stats["largest_score_drops"]:
                links = [f'<a href="#q{r["index"]}">#{r["index"]}（{r["score_change_pp"]:+.2f} 分）</a>'
                         for r in stats["largest_score_drops"]]
                output.append('<p class="detail">按官方分差选出的降幅最大题目　'+'　'.join(links)+'</p>')
            output.append(table([(k.upper(),v) for k,v in parts.items() if v["paired"]]+[(mode_labels[k],v) for k,v in s["by_dataset_and_composition"][d].items() if v["paired"]])+"</details>")
    output.append('<h2 id="answers">逐题完整回答</h2>')
    if not rows:
        output.append('<p>正在等待首个已提交的配对结果。</p>')
    for row in rows:
        output.append(case(row))
    output.append("<script>function openCase(){const e=document.getElementById(location.hash.slice(1));if(e&&e.tagName==='DETAILS'){e.open=true;e.scrollIntoView()}}addEventListener('hashchange',openCase);openCase();</script></main></body></html>")
    return "".join(output)


def render_report(rows, output_dir, *, expected_counts=None, mode="inference", score_fn=None):
    """Write index.html, summary.json and a complete re-readable answers.jsonl; return summary."""
    expected=expectations(expected_counts)
    data=validate_rows(rows,expected_counts=expected,score_fn=score_fn)
    require(isinstance(mode,str) and mode, "Specify an inference or reference mode")
    summary=build_summary(data,expected,mode)
    def encode(value):
        return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    out=Path(output_dir)
    write_atomic(out/"answers.jsonl","".join(encode(row)+"\n" for row in data))
    write_atomic(out/"summary.json",json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    write_atomic(out/"index.html",html_page(data,summary))
    return summary
