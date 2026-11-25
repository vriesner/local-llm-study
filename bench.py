"""
Benchmark a local (or cloud) model on a meeting-action-item extraction task.
One script, four backends: Ollama, llama.cpp (llama-server), LM Studio, OpenRouter.

Usage (PowerShell):
    py -m pip install psutil          # one-time, enables RAM tracking
    py bench.py qwen3.5:4b --backend ollama
    py bench.py qwen3.5-4b --backend llamacpp --label llamacpp_q4km
    py bench.py <id from http://localhost:1234/v1/models> --backend lmstudio
    $env:OPENROUTER_API_KEY="sk-or-..."
    py bench.py <openrouter/model-id> --backend openrouter

Expects data/test.jsonl (one JSON object per line):
{"id": 1, "transcript": "...", "expected": [{"owner": "Sam", "due": "Friday"}]}

Writes results/<label>_<timestamp>.csv (every request) and
results/<label>_<timestamp>_summary.json (the numbers + computer specs for your report).

Every backend is measured the same way: streaming, client-side time to first token,
and the same JSON-schema constrained output. Server-reported numbers are recorded
alongside where the server provides them.
"""
import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import threading
import time
import urllib.request

try:
    import psutil
except ImportError:
    psutil = None

SYSTEM_PROMPT = (
    "You extract action items from meeting transcripts. "
    'Return ONLY JSON in this exact shape: '
    '{"action_items": [{"owner": "<person>", "task": "<short task>", "due": "<due date or null>"}]}. '
    "If there are no action items, return {\"action_items\": []}."
)

SCHEMA = {
    "type": "object",
    "properties": {"action_items": {"type": "array", "items": {
        "type": "object",
        "properties": {"owner": {"type": "string"}, "task": {"type": "string"},
                       "due": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["owner", "task", "due"]}}},
    "required": ["action_items"],
}

BACKENDS = {  # root URL, process-name substrings for RAM tracking
    "ollama": ("http://localhost:11434", ["ollama"]),
    "llamacpp": ("http://localhost:8080", ["llama-server"]),
    "lmstudio": ("http://localhost:1234", ["lm studio", "llmworker"]),
    "openrouter": ("https://openrouter.ai/api", []),
    "custom": ("http://localhost:8000", []),
}


# ---------- Requests (all streaming) ----------
def call_ollama(base, model, transcript, num_ctx, use_schema):
    body = {
        "model": model, "stream": True, "think": False,
        "format": SCHEMA if use_schema else "json",
        "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": transcript}],
    }
    req = urllib.request.Request(base + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first, text, final, thinking = None, [], {}, False
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            if msg.get("thinking"):
                thinking = True
            piece = msg.get("content") or ""
            if (piece or msg.get("thinking")) and t_first is None:
                t_first = time.perf_counter()
            text.append(piece)
            if chunk.get("done"):
                final = chunk
    t_end = time.perf_counter()
    ev_s = final.get("eval_duration", 0) / 1e9
    return {
        "text": "".join(text), "ttft": (t_first or t_end) - t0,
        "decode_s": t_end - (t_first or t0), "wall": t_end - t0,
        "completion_tokens": final.get("eval_count", 0),
        "prompt_tokens": final.get("prompt_eval_count", 0),
        "thinking": thinking,
        "server_tps": round(final.get("eval_count", 0) / ev_s, 2) if ev_s else None,
        "server_ttft": round((final.get("load_duration", 0) + final.get("prompt_eval_duration", 0)) / 1e9, 3),
    }


def call_openai(base, key, model, transcript, use_schema, extra, backend):
    body = {
        "model": model, "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0, "seed": 0,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": transcript}],
    }
    if use_schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "action_items", "strict": True, "schema": SCHEMA}}
    if backend == "llamacpp":  # turn Qwen thinking off (needs --jinja on the server)
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body.update(extra)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    t_first, text, usage, chunks, thinking, timings = None, [], {}, 0, False, {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                chunk = json.loads(line[5:])
            except Exception:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]  # llama.cpp server-side numbers
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                reasoning = d.get("reasoning_content") or d.get("reasoning")
                if reasoning:
                    thinking = True
                piece = d.get("content") or ""
                if piece or reasoning:
                    chunks += 1
                    if t_first is None:
                        t_first = time.perf_counter()
                text.append(piece)
    t_end = time.perf_counter()
    out = "".join(text)
    return {
        "text": out, "ttft": (t_first or t_end) - t0,
        "decode_s": t_end - (t_first or t0), "wall": t_end - t0,
        "completion_tokens": usage.get("completion_tokens", chunks),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "thinking": thinking or "<think>" in out,
        "server_tps": round(timings["predicted_per_second"], 2) if timings.get("predicted_per_second") else None,
        "server_ttft": round(timings["prompt_ms"] / 1000, 3) if timings.get("prompt_ms") else None,
    }


def get_json(url, body=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers=headers or {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def server_info(backend, base, model):
    """Best-effort details about the engine and loaded model."""
    info = {}
    try:
        if backend == "ollama":
            info["engine_version"] = get_json(base + "/api/version").get("version")
            for m in get_json(base + "/api/ps").get("models", []):
                if model in (m.get("name"), m.get("model")):
                    info.update(model_total_gb=round(m["size"] / 1e9, 2),
                                model_in_vram_gb=round(m["size_vram"] / 1e9, 2),
                                context_length=m.get("context_length"))
            det = get_json(base + "/api/show", {"model": model}).get("details", {})
            info.update(quantization=det.get("quantization_level"),
                        parameter_size=det.get("parameter_size"))
        elif backend == "llamacpp":
            props = get_json(base + "/props")
            info["engine_version"] = props.get("build_info")
            info["context_length"] = props.get("default_generation_settings", {}).get("n_ctx")
            info["model_path"] = props.get("model_path")
    except Exception:
        pass
    return info


# ---------- Computer specs ----------
def _run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def _ps_json(query):
    out = _run(["powershell", "-NoProfile", "-Command", query + " | ConvertTo-Json -Compress"])
    try:
        data = json.loads(out)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def system_info():
    info = {}
    if platform.system() == "Windows":
        os_ = (_ps_json("Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber") or [{}])[0]
        cpus = _ps_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors")
        cs = (_ps_json("Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,Manufacturer,Model") or [{}])[0]
        gpus = _ps_json("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion")
        info["os_name"] = os_.get("Caption")
        info["os_version"] = f"{os_.get('Version')} (build {os_.get('BuildNumber')})"
        info["processors"] = [{"name": (c.get("Name") or "").strip(), "cores": c.get("NumberOfCores"),
                               "logical_processors": c.get("NumberOfLogicalProcessors")} for c in cpus]
        if cs.get("TotalPhysicalMemory"):
            info["ram_gb"] = round(int(cs["TotalPhysicalMemory"]) / 1024 ** 3, 1)
        info["computer"] = f"{cs.get('Manufacturer', '')} {cs.get('Model', '')}".strip()
        info["gpus"] = [{"name": g.get("Name"), "driver": g.get("DriverVersion")} for g in gpus]
    else:
        info["os_name"] = platform.system()
        info["os_version"] = platform.version()
        cpu = platform.processor()
        if platform.system() == "Linux":
            m = re.search(r"model name\s*:\s*(.+)", _run(["cat", "/proc/cpuinfo"]))
            cpu = m.group(1) if m else cpu
            gpu_lines = [l for l in _run(["lspci"]).splitlines() if re.search(r"VGA|3D|Display", l)]
            info["gpus"] = [{"name": l.split(": ", 1)[-1]} for l in gpu_lines]
        elif platform.system() == "Darwin":
            cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
            info["gpus"] = [{"name": n.strip()} for n in re.findall(r"Chipset Model:\s*(.+)", _run(["system_profiler", "SPDisplaysDataType"]))]
        info["processors"] = [{"name": cpu, "cores": psutil.cpu_count(logical=False) if psutil else None,
                               "logical_processors": os.cpu_count()}]
        if psutil:
            info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    if "ram_gb" not in info and psutil:
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    # NVIDIA GPUs: add accurate VRAM (Windows' generic adapter list under-reports it)
    nv = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
               "--format=csv,noheader,nounits"])
    info["nvidia_gpus"] = []
    for line in nv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            info["nvidia_gpus"].append({"index": int(parts[0]), "name": parts[1],
                                        "vram_mb": int(parts[2]), "driver": parts[3]})
    return info


# ---------- Memory sampling ----------
def gpu_used_list():
    """MiB used on each NVIDIA GPU, or [] if nvidia-smi is unavailable."""
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=5)
    try:
        return [int(x) for x in out.splitlines() if x.strip()]
    except ValueError:
        return []


class MemorySampler(threading.Thread):
    def __init__(self, procs, interval=0.5):
        super().__init__(daemon=True)
        self.procs, self.interval, self.stop_flag = procs, interval, threading.Event()
        self.peak_gpu, self.peak_proc_ram, self.peak_sys_ram = [], None, None

    def proc_ram_mb(self):
        if psutil is None or not self.procs:
            return None
        total = 0
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                if any(n in (p.info["name"] or "").lower() for n in self.procs):
                    total += p.info["memory_info"].rss
            except Exception:
                pass
        return total / 1e6

    def run(self):
        while not self.stop_flag.is_set():
            g = gpu_used_list()
            for i, v in enumerate(g):
                if i >= len(self.peak_gpu):
                    self.peak_gpu.append(v)
                else:
                    self.peak_gpu[i] = max(self.peak_gpu[i], v)
            r = self.proc_ram_mb()
            if r is not None:
                self.peak_proc_ram = max(self.peak_proc_ram or 0, r)
            if psutil:
                u = psutil.virtual_memory().used / 1e6
                self.peak_sys_ram = max(self.peak_sys_ram or 0, u)
            time.sleep(self.interval)


# ---------- Scoring ----------
def norm(s):
    return str(s or "").strip().lower()


def norm_due(s):
    """Lowercase and drop leading filler words so 'by Friday' == 'the 15th' == '15th'."""
    s = norm(s)
    changed = True
    while changed:
        changed = False
        for prefix in ("by ", "on ", "before ", "until ", "at ", "in ", "the "):
            if s.startswith(prefix):
                s, changed = s[len(prefix):], True
    return s if s and s != "none" else "null"


def score(output_text, expected):
    try:
        items = json.loads(output_text)["action_items"]
        pred = {(norm(i.get("owner")), norm_due(i.get("due"))) for i in items}
    except Exception:
        return False, 0.0, 0.0, 0.0
    gold = {(norm(e["owner"]), norm_due(e.get("due"))) for e in expected}
    if not gold and not pred:
        return True, 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return True, p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model name as the server knows it")
    ap.add_argument("--backend", choices=list(BACKENDS), default="ollama")
    ap.add_argument("--base-url", help="override the server root URL")
    ap.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    ap.add_argument("--label", help="used in output file names (default: backend_model)")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--num_ctx", type=int, default=4096,
                    help="applied per-request for Ollama; for others set it when starting the server")
    ap.add_argument("--no-schema", action="store_true", help="plain JSON mode instead of a JSON schema")
    ap.add_argument("--proc", nargs="*", help="process-name substrings for RAM tracking")
    ap.add_argument("--extra", default="{}", help="extra JSON merged into OpenAI-style request bodies")
    ap.add_argument("--notes", default="", help="free text saved in the summary (e.g. which GPU was used)")
    args = ap.parse_args()

    root, default_procs = BACKENDS[args.backend]
    base = (args.base_url or root).rstrip("/")
    procs = [p.lower() for p in (args.proc if args.proc is not None else default_procs)]
    extra = json.loads(args.extra)
    use_schema = not args.no_schema
    label = args.label or f"{args.backend}_{args.model}"
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)

    def call(transcript):
        if args.backend == "ollama":
            return call_ollama(base, args.model, transcript, args.num_ctx, use_schema)
        return call_openai(base, args.api_key, args.model, transcript, use_schema, extra, args.backend)

    with open(args.data, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if psutil is None:
        print("Note: psutil not installed, so RAM won't be tracked (run: py -m pip install psutil)")

    print("Collecting computer specs...")
    sysinfo = system_info()
    baseline_gpu = gpu_used_list()
    baseline_ram = round(psutil.virtual_memory().used / 1e6) if psutil else None

    print(f"Loaded {len(examples)} examples. Warming up {args.model} on {args.backend}...")
    t_warm = time.perf_counter()
    call(examples[0]["transcript"])  # not recorded
    cold_start_s = time.perf_counter() - t_warm
    info = server_info(args.backend, base, args.model)

    sampler = MemorySampler(procs)
    sampler.start()
    rows, thinking_hits = [], 0
    t_start = time.perf_counter()
    for run in range(1, args.runs + 1):
        for ex in examples:
            r = call(ex["transcript"])
            thinking_hits += bool(r["thinking"])
            valid, p, rc, f1 = score(r["text"], ex["expected"])
            rows.append({
                "backend": args.backend, "model": args.model, "run": run, "id": ex["id"],
                "valid_json": int(valid), "precision": p, "recall": rc, "f1": f1,
                "tokens_per_s": round(r["completion_tokens"] / r["decode_s"], 2) if r["decode_s"] > 0 else 0.0,
                "ttft_s": round(r["ttft"], 3),
                "server_tokens_per_s": r["server_tps"], "server_prompt_s": r["server_ttft"],
                "prompt_tokens": r["prompt_tokens"], "output_tokens": r["completion_tokens"],
                "wall_s": round(r["wall"], 3), "output": r["text"],
            })
        print(f"  run {run}/{args.runs} done")
    total_s = time.perf_counter() - t_start
    sampler.stop_flag.set()
    sampler.join(timeout=3)

    os.makedirs("results", exist_ok=True)
    base_name = f"results/{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    with open(base_name + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    first = [r for r in rows if r["run"] == 1]  # accuracy is deterministic (temp 0, seed 0)
    summary = {
        "backend": args.backend, "model": args.model, "label": label,
        "examples": len(examples), "runs": args.runs,
        "valid_json_rate": round(sum(r["valid_json"] for r in first) / len(first), 4),
        "mean_f1": round(statistics.mean(r["f1"] for r in first), 4),
        "median_tokens_per_s": round(statistics.median(r["tokens_per_s"] for r in rows), 2),
        "median_ttft_s": round(statistics.median(r["ttft_s"] for r in rows), 3),
        "median_seconds_per_request": round(statistics.median(r["wall_s"] for r in rows), 2),
        "total_benchmark_time_s": round(total_s, 1),
        "cold_start_s": round(cold_start_s, 2),
        "num_ctx_requested": args.num_ctx,
        "structured_output": "json_schema" if use_schema else "json_mode",
        "requests_with_thinking_output": int(thinking_hits),
        "gpu_used_mb_before": baseline_gpu,
        "peak_gpu_used_mb_per_gpu": sampler.peak_gpu,
        "system_ram_used_mb_before": baseline_ram,
        "peak_system_ram_used_mb": round(sampler.peak_sys_ram) if sampler.peak_sys_ram else None,
        "peak_server_process_ram_mb": round(sampler.peak_proc_ram) if sampler.peak_proc_ram else None,
        **info,
        "notes": args.notes,
        "system": sysinfo,
    }
    with open(base_name + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY:", label, "===")
    print(f"Valid JSON rate        : {summary['valid_json_rate']:.1%}")
    print(f"Mean F1                : {summary['mean_f1']:.3f}")
    print(f"Median tokens/s        : {summary['median_tokens_per_s']}")
    print(f"Median TTFT (s)        : {summary['median_ttft_s']}")
    print(f"Median s per request   : {summary['median_seconds_per_request']}")
    print(f"Total benchmark time   : {total_s:.1f} s ({total_s / 60:.1f} min) for {len(rows)} requests")
    print(f"Cold start (s)         : {cold_start_s:.1f}")
    print(f"GPU memory (MB)        : {baseline_gpu} before -> {sampler.peak_gpu} peak (per GPU)")
    print(f"System RAM used (MB)   : {baseline_ram} before -> {summary['peak_system_ram_used_mb']} peak")
    if info:
        print("Server info            :", {k: v for k, v in info.items()})
    cpu = ", ".join(c["name"] for c in sysinfo.get("processors", []))
    gpus = ", ".join(g["name"] for g in sysinfo.get("gpus", []) if g.get("name"))
    print(f"Computer               : {sysinfo.get('os_name')} {sysinfo.get('os_version')} | {cpu} | "
          f"{sysinfo.get('ram_gb')} GB RAM | GPU(s): {gpus}")
    if thinking_hits:
        print(f"\nWARNING: thinking output seen in {thinking_hits} requests. Disable thinking for a fair comparison.")
    print(f"Saved: {base_name}.csv and {base_name}_summary.json")


if __name__ == "__main__":
    main()
