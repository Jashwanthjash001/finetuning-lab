"""Generate the tiny structured-extraction dataset (deterministic, seed 0).

200 synthetic sentences -> {name, company, role} JSON. 170 train / 30 valid.
Same data for the LoRA and QLoRA runs so the comparison is fair.
"""
import json, random
from pathlib import Path

random.seed(0)

names = ["Priya Sharma", "John Carter", "Aisha Khan", "Diego Alvarez", "Mei Lin", "Rahul Nair",
         "Sara Okoro", "Tom Becker", "Lena Petrova", "Arjun Rao", "Fatima Noor", "Chris Evans"]
companies = ["Acme Corp", "Globex", "Initech", "Umbrella Labs", "Stark Industries", "Hooli",
             "Pied Piper", "Nimbus AI", "Vertex Systems", "Orbital Dynamics"]
roles = ["data scientist", "backend engineer", "product manager", "ML engineer",
         "DevOps engineer", "research scientist", "software engineer", "solutions architect"]
templates = [
    "{n} joined {c} last year as a {r}.",
    "At {c}, {n} works as a {r}.",
    "{n} is a {r} at {c}.",
    "We hired {n} as a {r} for {c} in 2024.",
    "{c} recently welcomed {n}, their new {r}.",
    "Last quarter, {c} promoted {n} to {r}.",
]


def make_example():
    n, c, r = random.choice(names), random.choice(companies), random.choice(roles)
    text = random.choice(templates).format(n=n, c=c, r=r)
    user = ("Extract name, company, and role from the text and return ONLY JSON "
            f"with keys name, company, role.\n\nText: {text}")
    assistant = json.dumps({"name": n, "company": c, "role": r})
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def main():
    data = [make_example() for _ in range(200)]
    train, valid = data[:170], data[170:]
    Path("data").mkdir(exist_ok=True)
    with open("data/train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open("data/valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(ex) + "\n")
    print(f"wrote data/train.jsonl ({len(train)}) and data/valid.jsonl ({len(valid)})")


if __name__ == "__main__":
    main()
