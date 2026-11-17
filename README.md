# LoRA Fine-Tuning Lab — Llama 3.2 1B on Apple Silicon

A hands-on notebook that LoRA-fine-tunes **Llama 3.2 1B** locally on a Mac (MLX, no cloud) for a
small structured-extraction task, and measures the before/after. The point isn't the task — it's
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

## How it works

- **LoRA** — freeze the base model, train a small adapter (<1% of params). The notebook explains
  LoRA vs QLoRA and when you'd reach for each.
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

## Honest limitations

- Tiny **synthetic** dataset and a **1B** model on a single narrow task — this is a learning /
  demonstration project, not a production fine-tune.
- Exact-match is a strict but shallow metric; it proves the model learned the format, not broad
  capability.
- Apple Silicon only (MLX).
