"""LoRA vs QLoRA on Llama 3.2 1B (MLX), same data and same config.

QLoRA here = train the LoRA adapter on a 4-bit-quantized base instead of the bf16 base.
Everything else (data, iters, batch, layers, lr) is identical, so the comparison is fair.

Measures, for each run: peak process memory, wall-clock train time, base size on disk,
and exact-match on the held-out set before/after. Writes results/lora_vs_qlora.json.

Run with the clean MLX venv:
  .venv-mlx/bin/python lora_vs_qlora.py
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv-mlx/bin/python")
BF16 = "mlx-community/Llama-3.2-1B-Instruct-bf16"
Q4 = str(ROOT / "models/llama32-1b-q4")
RESULTS = ROOT / "results"

# identical training config for both runs
CFG = dict(iters="200", batch="2", layers="8", lr="1e-4")
SCORE_N = "20"


def sh(cmd: list[str], capture=False) -> str:
    """Run a command, stream it live, return combined output. Raise on failure."""
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    buf = []
    for line in p.stdout:
        print(line, end="", flush=True)
        buf.append(line)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"failed (exit {p.returncode}): {' '.join(cmd[:4])}")
    return "".join(buf)


def train(model: str, adapter: str) -> dict:
    """Train a LoRA adapter on `model`; capture peak RSS (bytes) and wall seconds."""
    cmd = ["/usr/bin/time", "-l", PY, "-m", "mlx_lm.lora",
           "--model", model, "--train", "--data", "data",
           "--iters", CFG["iters"], "--batch-size", CFG["batch"],
           "--num-layers", CFG["layers"], "--learning-rate", CFG["lr"],
           "--adapter-path", adapter, "--steps-per-report", "20", "--steps-per-eval", "100"]
    t0 = time.monotonic()
    out = sh(cmd)
    wall = time.monotonic() - t0
    m = re.search(r"([0-9]+)\s+maximum resident set size", out)
    peak = int(m.group(1)) if m else 0
    return {"wall_s": round(wall, 1), "peak_rss_gb": round(peak / 1e9, 2)}


def score(model: str, adapter: str | None) -> float:
    cmd = [PY, "mlx_tasks.py", "score", "--model", model, "--n", SCORE_N]
    if adapter:
        cmd += ["--adapter", adapter]
    out = sh(cmd)
    return float(re.search(r"EXACT_MATCH_RATE ([\d.]+)", out).group(1))


def dir_size_gb(path: str) -> float:
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / 1e9, 2)


def hf_cache_dir(repo: str) -> str:
    base = Path.home() / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--"))
    snaps = list((base / "snapshots").glob("*"))
    return str(snaps[0]) if snaps else str(base)


def main():
    os.chdir(ROOT)
    RESULTS.mkdir(exist_ok=True)

    # 0. data (deterministic; regenerate only if missing)
    if not Path("data/train.jsonl").exists():
        sh([PY, "make_data.py"])

    # 1. quantize base to 4-bit (the QLoRA base) -- skip if already done
    if not Path(Q4, "config.json").exists():
        sh([PY, "-m", "mlx_lm.convert", "--hf-path", BF16, "--mlx-path", Q4, "-q"])

    res: dict = {"config": CFG, "score_n": int(SCORE_N)}

    # 2. base sizes on disk
    res["base_size_gb"] = {
        "bf16": dir_size_gb(hf_cache_dir(BF16)),
        "q4": dir_size_gb(Q4),
    }

    # 3. LoRA (bf16 base)
    print("\n########## LoRA (bf16 base) ##########")
    res["lora"] = train(BF16, "adapters-lora")
    res["lora"]["exact_before"] = score(BF16, None)
    res["lora"]["exact_after"] = score(BF16, "adapters-lora")

    # 4. QLoRA (4-bit base)
    print("\n########## QLoRA (4-bit base) ##########")
    res["qlora"] = train(Q4, "adapters-qlora")
    res["qlora"]["exact_before"] = score(Q4, None)
    res["qlora"]["exact_after"] = score(Q4, "adapters-qlora")

    (RESULTS / "lora_vs_qlora.json").write_text(json.dumps(res, indent=2))

    # 5. summary
    L, Q = res["lora"], res["qlora"]
    print("\n\n==================== SUMMARY ====================")
    print(f"base size on disk   LoRA(bf16) {res['base_size_gb']['bf16']} GB   "
          f"QLoRA(q4) {res['base_size_gb']['q4']} GB")
    print(f"peak train memory   LoRA {L['peak_rss_gb']} GB   QLoRA {Q['peak_rss_gb']} GB")
    print(f"train wall time     LoRA {L['wall_s']}s   QLoRA {Q['wall_s']}s")
    print(f"exact-match before  LoRA {L['exact_before']:.0%}   QLoRA {Q['exact_before']:.0%}")
    print(f"exact-match after   LoRA {L['exact_after']:.0%}   QLoRA {Q['exact_after']:.0%}")
    print("wrote results/lora_vs_qlora.json")


if __name__ == "__main__":
    main()
