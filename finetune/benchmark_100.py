"""Benchmark 100 queries — mix of full words, partial words, and short prefixes."""

import json
import random
import urllib.request
import sys

URL = os.environ.get("AUTOCOMPLETE_URL", "http://localhost:8400/autocomplete")

# Load real queries from training data
all_queries = set()
for line in open("/Users/ron/Desktop/tipstat/sourcer/autocomplete/finetune/train.jsonl"):
    line = line.strip()
    if not line:
        continue
    try:
        t = json.loads(line)["text"]
        words = t.split()
        if 3 <= len(words) <= 10:
            all_queries.add(t)
    except:
        pass

random.seed(42)
source_queries = random.sample(sorted(all_queries), 100)

# Build test cases: mix of full prefix, partial word, and short prefix
test_cases = []
for i, q in enumerate(source_queries):
    words = q.split()
    if i % 3 == 0:
        # Full words — take first 1-2 words
        n = min(2, len(words))
        test_cases.append((" ".join(words[:n]), "full"))
    elif i % 3 == 1:
        # Partial word — take first word + partial second word
        if len(words) >= 2 and len(words[1]) >= 3:
            partial = words[1][: len(words[1]) // 2 + 1]
            test_cases.append((words[0] + " " + partial, "partial"))
        else:
            test_cases.append((words[0], "full"))
    else:
        # Longer prefix — take first 3 words
        n = min(3, len(words))
        test_cases.append((" ".join(words[:n]), "long"))

good = 0
bad = 0
total_suggestions = 0
total_latency = 0
hinglish_count = 0
garbled_count = 0
results_by_type = {"full": [0, 0], "partial": [0, 0], "long": [0, 0]}

HINDI_WORDS = {
    "ka", "ke", "ki", "ko", "kya", "hai", "hain", "mein", "se", "ya",
    "konsa", "chahiye", "chahta", "kaise", "kahan", "kaisa", "karna",
    "dikhao", "batao", "liye", "raha", "hoon", "nahi", "bhi", "aur",
    "wala", "wale", "koi", "loon", "milta", "milte", "doon", "hoga",
}


def has_hinglish(text):
    words = text.lower().split()
    count = sum(1 for w in words if w in HINDI_WORDS)
    return count >= 2


def is_garbled(text):
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return len(text) > 0 and ascii_count / len(text) < 0.85


for i, (query, qtype) in enumerate(test_cases):
    try:
        data = json.dumps({"text": query}).encode()
        req = urllib.request.Request(
            URL, data=data, headers={"Content-Type": "application/json"}
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
        suggestions = resp.get("suggestions", [])
    except Exception as e:
        print(f"  [{i+1}] ERROR: {query} — {e}")
        bad += 1
        results_by_type[qtype][1] += 1
        continue

    has_results = len(suggestions) > 0
    if has_results:
        good += 1
        results_by_type[qtype][0] += 1
    else:
        bad += 1
        results_by_type[qtype][1] += 1

    for s in suggestions:
        total_suggestions += 1
        total_latency += s["latency_ms"]
        if has_hinglish(s["full_text"]):
            hinglish_count += 1
        if is_garbled(s["full_text"]):
            garbled_count += 1

    # Print every query with results
    status = "OK" if has_results else "EMPTY"
    print(f"  [{i+1:3d}] [{qtype:7s}] [{status:5s}] \"{query}\"")
    if has_results:
        for s in suggestions[:3]:
            print(f"         -> {s['full_text']}  ({s['latency_ms']:.0f}ms)")

print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")
print(f"Queries with results:  {good}/100 ({good}%)")
print(f"Empty results:         {bad}/100")
print(f"Total suggestions:     {total_suggestions}")
print(f"Avg suggestions/query: {total_suggestions/max(good,1):.1f}")
print(f"Avg latency:           {total_latency/max(total_suggestions,1):.0f}ms")
print(f"Hinglish suggestions:  {hinglish_count}/{total_suggestions} ({100*hinglish_count/max(total_suggestions,1):.1f}%)")
print(f"Garbled suggestions:   {garbled_count}/{total_suggestions} ({100*garbled_count/max(total_suggestions,1):.1f}%)")
print(f"\nBy query type:")
for qtype in ["full", "partial", "long"]:
    ok, fail = results_by_type[qtype]
    total = ok + fail
    print(f"  {qtype:8s}: {ok}/{total} returned results ({100*ok/max(total,1):.0f}%)")
