"""Benchmark 20 B2B queries against the autocomplete service."""

import json
import os
import sys
import urllib.request

URL = os.environ.get("AUTOCOMPLETE_URL", "http://localhost:8400/autocomplete")

QUERIES = [
    "dental chair",
    "stainless steel",
    "solar panel",
    "tmt bar",
    "cement",
    "hydraulic press",
    "copper wire",
    "led street",
    "water pump",
    "conveyor belt",
    "packaging mac",
    "ball bearing",
    "welding",
    "forklift",
    "safety helmet",
    "pvc pipe",
    "diesel gen",
    "cnc machine",
    "transformer",
    "air compressor",
]

good = 0
bad = 0
total_suggestions = 0

for q in QUERIES:
    data = json.dumps({"text": q}).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    suggestions = resp.get("suggestions", [])
    print(f"\n=== {q} ===")
    if not suggestions:
        print("  [no suggestions]")
        bad += 1
        continue
    for s in suggestions:
        ft = s["full_text"]
        ms = s["latency_ms"]
        print(f"  {ft}  ({ms:.0f}ms)")
        total_suggestions += 1
    good += 1

print(f"\n{'='*60}")
print(f"Queries with results: {good}/{len(QUERIES)}")
print(f"Total suggestions: {total_suggestions}")
print(f"Avg suggestions per query: {total_suggestions/max(good,1):.1f}")
