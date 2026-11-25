"""
Benchmark a local Ollama model on a meeting-action-item extraction task.

Usage (PowerShell):
    py -m pip install psutil          # one-time, enables RAM tracking
    py bench.py qwen3.5:4b
    py bench.py qwen3.5:4b-q8_0 --runs 3

Expects data/test.jsonl (one JSON object per line):
{"id": 1, "transcript": "...", "expected": [{"owner": "Sam", "due": "Friday"}]}

Writes results/<model>_<timestamp>.csv (every request) and
results/<model>_<timestamp>_summary.json (the numbers for your report).
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request

try:
    import psutil
except ImportError:
    psutil = None

BASE = "http://localhost:11434"
SYSTEM_PROMPT = (
    "You extract action items from meeting transcripts. "
    'Return ONLY JSON in this exact shape: '
    '{"action_items": [{"owner": "<person>", "task": "<short task>", "due": "<due date or null>"}]}. '
    "If there are no action items, return {\"action_items\": []}."
)


# ---------- Ollama calls ----------
def call_ollama(model, transcript, num_ctx):
    body = {
        "model": model, "stream": False, "think": False, "format": "json",
        "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": transcript}],
    }
    req = urllib.request.Request(BASE + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    return data, time.perf_counter() - t0


def loaded_model_info(model):
    """Ask Ollama how much memory the loaded model uses and how much is on the GPU."""
    try:
        with urllib.request.urlopen(BASE + "/api/ps", timeout=10) as resp:
            for m in json.loads(resp.read().decode()).get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    return {"model_total_gb": round(m["size"] / 1e9, 2),
                            "model_in_vram_gb": round(m["size_vram"] / 1e9, 2),
                            "context_length": m.get("context_length")}
    except Exception:
        pass
    return {}


# ---------- Memory sampling ----------
def gpu_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return max(int(x) for x in out) if out else None
    except Exception:
        return None


def ollama_ram_mb():
    if psutil is None:
        return None
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            if "ollama" in (p.info["name"] or "").lower():
                total += p.info["memory_info"].rss
        except Exception:
            pass
    return total / 1e6


class MemorySampler(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval, self.stop_flag = interval, threading.Event()
        self.peak_gpu, self.peak_ram = None, None

    def run(self):
        while not self.stop_flag.is_set():
            g, r = gpu_used_mb(), ollama_ram_mb()
            if g is not None:
                self.peak_gpu = max(self.peak_gpu or 0, g)
            if r is not None:
                self.peak_ram = max(self.peak_ram or 0, r)
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
    ap.add_argument("model")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--num_ctx", type=int, default=4096)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if psutil is None:
        print("Note: psutil not installed, so Ollama RAM won't be tracked "
              "(run: py -m pip install psutil)")

    baseline_gpu = gpu_used_mb()
    print(f"Loaded {len(examples)} examples. Warming up {args.model}...")
    t_warm = time.perf_counter()
    call_ollama(args.model, examples[0]["transcript"], args.num_ctx)  # not recorded
    cold_start_s = time.perf_counter() - t_warm
    mem_info = loaded_model_info(args.model)

    sampler = MemorySampler()
    sampler.start()
    rows = []
    t_start = time.perf_counter()
    for run in range(1, args.runs + 1):
        for ex in examples:
            data, wall = call_ollama(args.model, ex["transcript"], args.num_ctx)
            out = data["message"]["content"]
            valid, p, r, f1 = score(out, ex["expected"])
            eval_s = data.get("eval_duration", 0) / 1e9
            rows.append({
                "model": args.model, "run": run, "id": ex["id"],
                "valid_json": int(valid), "precision": p, "recall": r, "f1": f1,
                "tokens_per_s": round(data.get("eval_count", 0) / eval_s, 2) if eval_s else 0.0,
                "ttft_s": round((data.get("load_duration", 0) + data.get("prompt_eval_duration", 0)) / 1e9, 3),
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
                "wall_s": round(wall, 3), "output": out,
            })
        print(f"  run {run}/{args.runs} done")
    total_s = time.perf_counter() - t_start
    sampler.stop_flag.set()
    sampler.join(timeout=3)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"results/{args.model.replace(':', '_').replace('/', '_')}_{stamp}"
    os.makedirs("results", exist_ok=True)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    first = [r for r in rows if r["run"] == 1]  # accuracy is deterministic (temp 0, seed 0)
    summary = {
        "model": args.model, "examples": len(examples), "runs": args.runs,
        "valid_json_rate": round(sum(r["valid_json"] for r in first) / len(first), 4),
        "mean_f1": round(statistics.mean(r["f1"] for r in first), 4),
        "median_tokens_per_s": round(statistics.median(r["tokens_per_s"] for r in rows), 2),
        "median_ttft_s": round(statistics.median(r["ttft_s"] for r in rows), 3),
        "median_seconds_per_request": round(statistics.median(r["wall_s"] for r in rows), 2),
        "total_benchmark_time_s": round(total_s, 1),
        "cold_start_s": round(cold_start_s, 2),
        "num_ctx": args.num_ctx,
        "gpu_used_mb_before": baseline_gpu,
        "peak_gpu_used_mb_system": sampler.peak_gpu,
        "peak_ollama_ram_mb": round(sampler.peak_ram) if sampler.peak_ram else None,
        **mem_info,
    }
    with open(base + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY:", args.model, "===")
    print(f"Valid JSON rate        : {summary['valid_json_rate']:.1%}")
    print(f"Mean F1                : {summary['mean_f1']:.3f}")
    print(f"Median tokens/s        : {summary['median_tokens_per_s']}")
    print(f"Median TTFT (s)        : {summary['median_ttft_s']}")
    print(f"Median s per request   : {summary['median_seconds_per_request']}")
    print(f"Total benchmark time   : {total_s:.1f} s ({total_s / 60:.1f} min) for {len(rows)} requests")
    print(f"Cold start (s)         : {cold_start_s:.1f}")
    if mem_info:
        print(f"Model memory           : {mem_info['model_total_gb']} GB total, "
              f"{mem_info['model_in_vram_gb']} GB on GPU")
    print(f"System GPU memory (MB) : {baseline_gpu} before -> {sampler.peak_gpu} peak")
    print(f"Peak Ollama RAM (MB)   : {summary['peak_ollama_ram_mb']}")
    print(f"Saved: {base}.csv and {base}_summary.json")


if __name__ == "__main__":
    main()
