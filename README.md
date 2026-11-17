# LoRA Fine-Tuning Lab — Llama 3.2 1B on Apple Silicon

A hands-on lab that **LoRA- and QLoRA-fine-tunes Llama 3.2 1B** locally on a Mac (MLX, no cloud) for
a small structured-extraction task, and measures the before/after. The point isn't the task — it's
seeing, end to end, *what fine-tuning actually changes* and proving it with numbers.

## The task & result

Teach the base model to read a sentence and return strict JSON: `{name, company, role}`, copying the
role **verbatim** (the base model likes to "tidy" `staff ML engineer` → `Staff ML Engineer`, which
breaks exact match).

| | Exact-match on held-out set |
| --- | --- |
| Base Llama 3.2 1B | ~15% |
| After LoRA (a few hundred steps) | ~100% |

Validation loss drops from ~4.5 → ~0.1 over training. Small task, but the gap is real and the
mechanism is visible.

## LoRA vs QLoRA

QLoRA just means training the same adapter on a **4-bit-quantized base** instead of the full bf16
one — same data, same config, only the frozen base changes. `lora_vs_qlora.py` runs both back to
back and measures the trade-off (M1 Pro, 200 iters):

| | Base on disk | Peak train memory | Train time | Exact-match (before → after) |
| --- | --- | --- | --- | --- |
| LoRA (bf16 base) | 2.48 GB | 2.10 GB | 89 s | 15% → 100% |
| QLoRA (4-bit base) | **0.71 GB** | **1.24 GB** | 112 s | 5% → 100% |

Quantizing the base cuts its footprint ~71% and peak training memory ~41%, and still reaches 100%
exact-match. You pay for it with ~25% slower training (4-bit matmul overhead) and a weaker starting
point (5% vs 15%) that the adapter fully recovers. On a 16 GB Mac, that memory saving is what lets
you fine-tune a base that otherwise wouldn't fit. Raw numbers: `results/lora_vs_qlora.json`.

## How it works

- **LoRA / QLoRA** — freeze the base model, train a small adapter (<1% of params). QLoRA freezes a
  4-bit-quantized base instead (see the table above). The notebook covers when you'd reach for each.
- **MLX** — Apple's on-device ML framework; training runs on the Mac's GPU via `mlx_lm.lora`.
- **Memory-safe by design** — every model load/generate runs in a short-lived subprocess
  (`mlx_tasks.py`), so a 16 GB Mac never holds two models at once and doesn't OOM.
- The notebook walks the full loop: install → baseline eval → build data → train → re-eval → fuse
  the adapter back into a standalone model.

## Run it

```
# open the notebook, restart the kernel, run top to bottom
jupyter lab lora_finetune_llama32_1b.ipynb
```

The setup cell builds a clean `.venv-mlx` from python.org Python (not Anaconda — Anaconda's bundled
MPI clashes with MLX). First run downloads the 1B model (~2 GB).

Or run the LoRA-vs-QLoRA comparison directly (quantizes the base, trains both, prints the table):

```
.venv-mlx/bin/python lora_vs_qlora.py
```

## Honest limitations

- Tiny **synthetic** dataset and a **1B** model on a single narrow task — this is a learning /
  demonstration project, not a production fine-tune.
- Exact-match is a strict but shallow metric; it proves the model learned the format, not broad
  capability.
- Apple Silicon only (MLX).
