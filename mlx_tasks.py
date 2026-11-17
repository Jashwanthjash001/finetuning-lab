"""Short-lived MLX tasks: load ONE model, do one thing, exit (frees memory).

Run with the clean venv's python so each call loads a single model and releases
it on exit -- this keeps 16 GB machines from OOM'ing when comparing two models.

  python mlx_tasks.py generate --model <repo> [--adapter adapters] --text "..."
  python mlx_tasks.py score    --model <repo> [--adapter adapters] [--data data/valid.jsonl] [--n 30]
"""
import argparse
import json
from mlx_lm import load, generate

PROMPT = ("Extract name, company, and role from the text and return ONLY JSON "
          "with keys name, company, role.\n\nText: {text}")


def parse_obj(text):
    """Best-effort parse of the first JSON object in the model output, else None."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    lo, hi = t.find("{"), t.rfind("}")
    if lo == -1 or hi == -1:
        return None
    try:
        obj = json.loads(t[lo:hi + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["generate", "score"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--text", default="")
    ap.add_argument("--data", default="data/valid.jsonl")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()

    model, tok = (load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model))

    if a.task == "generate":
        content = PROMPT.replace("{text}", a.text)   # safe even if text has braces
        msgs = [{"role": "user", "content": content}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        result = generate(model, tok, prompt=prompt, max_tokens=120, verbose=False)
        print(result.strip())
    else:
        with open(a.data, encoding="utf-8") as fh:
            examples = [json.loads(line) for line in fh][:a.n]
        exact = 0            # output JSON exactly equals the ground-truth object
        for ex in examples:
            user_msg, asst_msg = ex["messages"][0], ex["messages"][1]
            expected = json.loads(asst_msg["content"])
            prompt = tok.apply_chat_template([user_msg], add_generation_prompt=True, tokenize=False)
            out = generate(model, tok, prompt=prompt, max_tokens=80, verbose=False)
            exact += (parse_obj(out) == expected)
        rate = exact / len(examples) if examples else 0.0
        print(f"EXACT_MATCH_RATE {rate:.4f} over {len(examples)}")


if __name__ == "__main__":
    main()
