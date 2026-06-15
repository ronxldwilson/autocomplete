"""
Prepare training data for SmolLM2-135M autocomplete fine-tuning.

For each query, we create multiple training examples by splitting at different
word boundaries. This teaches the model to complete queries from any prefix.

Example query: "dental chair hydraulic with LED"
Generates:
  "dental" -> " chair hydraulic with LED"
  "dental chair" -> " hydraulic with LED"
  "dental chair hydraulic" -> " with LED"
  "dental chair hydraulic with" -> " LED"
"""

import json
import os
import random

DATASET_PATH = "/Users/ron/Desktop/tipstat/sourcer/dspy-enhancement/evals/datasets/intent_routing"
OUTPUT_DIR = "/Users/ron/Desktop/tipstat/sourcer/autocomplete/finetune"
SKIP_FILES = {"invalid.json", "safety_edge_cases.json", "mismatch_tier.json"}
MIN_QUERY_LEN = 8
MIN_PREFIX_WORDS = 1


def load_queries():
    queries = []
    for f in sorted(os.listdir(DATASET_PATH)):
        if not f.endswith(".json") or f in SKIP_FILES:
            continue
        data = json.load(open(os.path.join(DATASET_PATH, f)))
        examples = data.get("examples", []) if isinstance(data, dict) else data
        for item in examples:
            q = item.get("query", "") if isinstance(item, dict) else str(item)
            q = q.strip()
            if len(q) >= MIN_QUERY_LEN:
                queries.append(q)
    return list(set(queries))


def create_training_examples(queries):
    examples = []
    for query in queries:
        words = query.split()
        if len(words) < 2:
            continue
        for split_at in range(MIN_PREFIX_WORDS, len(words)):
            prefix = " ".join(words[:split_at])
            completion = " " + " ".join(words[split_at:])
            examples.append({"prefix": prefix, "completion": completion, "full": query})
    return examples


def write_train_file(examples, path):
    with open(path, "w") as f:
        for ex in examples:
            line = ex["prefix"] + ex["completion"]
            f.write(line + "\n")


def write_jsonl(examples, path):
    with open(path, "w") as f:
        for ex in examples:
            json.dump({"text": ex["prefix"] + ex["completion"]}, f)
            f.write("\n")


def main():
    random.seed(42)
    queries = load_queries()
    print(f"Loaded {len(queries)} unique queries")

    examples = create_training_examples(queries)
    random.shuffle(examples)
    print(f"Generated {len(examples)} training examples")

    split = int(len(examples) * 0.95)
    train = examples[:split]
    val = examples[split:]
    print(f"Train: {len(train)}, Val: {len(val)}")

    write_train_file(train, os.path.join(OUTPUT_DIR, "train.txt"))
    write_train_file(val, os.path.join(OUTPUT_DIR, "val.txt"))
    write_jsonl(train, os.path.join(OUTPUT_DIR, "train.jsonl"))
    write_jsonl(val, os.path.join(OUTPUT_DIR, "val.jsonl"))

    print(f"\nSample training lines:")
    for ex in random.sample(train, 5):
        print(f"  [{ex['prefix']}] -> [{ex['completion'].strip()}]")


if __name__ == "__main__":
    main()
