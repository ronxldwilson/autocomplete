"""Quick benchmark — 20 queries, report latency + quality stats."""

import json
import os
import urllib.request

URL = os.environ.get("AUTOCOMPLETE_URL", "http://localhost:8400/autocomplete")

QUERIES = [
    "dental chair", "stainless steel", "solar panel", "tmt bar", "cement",
    "hydraulic press", "copper wire", "led street", "water pump", "conveyor belt",
    "packaging mac", "ball bearing", "welding", "forklift", "safety helmet",
    "pvc pipe", "diesel gen", "cnc machine", "transformer", "air compressor",
]

HINDI_WORDS = {
    "ka", "ke", "ki", "ko", "kya", "hai", "hain", "mein", "se", "ya",
    "konsa", "chahiye", "chahta", "kaise", "kahan", "karna", "dikhao",
    "batao", "liye", "raha", "hoon", "nahi", "bhi", "koi", "milta",
}

good = 0
total_s = 0
total_lat = 0
hinglish = 0
garbled = 0
samples = []

for q in QUERIES:
    try:
        data = json.dumps({"text": q}).encode()
        req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
        suggestions = resp.get("suggestions", [])
    except:
        continue
    if suggestions:
        good += 1
    for s in suggestions:
        total_s += 1
        total_lat += s["latency_ms"]
        words = s["full_text"].lower().split()
        hc = sum(1 for w in words if w in HINDI_WORDS)
        if hc >= 2:
            hinglish += 1
        ascii_r = sum(1 for c in s["full_text"] if ord(c) < 128) / max(len(s["full_text"]), 1)
        if ascii_r < 0.85:
            garbled += 1
    if suggestions:
        samples.append(f"  {q}: {suggestions[0]['full_text']}  ({suggestions[0]['latency_ms']:.0f}ms)")

avg_lat = total_lat / max(total_s, 1)
print(f"Queries OK: {good}/20 | Suggestions: {total_s} | Avg latency: {avg_lat:.0f}ms | Hindi: {hinglish} | Garbled: {garbled}")
for s in samples[:10]:
    print(s)
