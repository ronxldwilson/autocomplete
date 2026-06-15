"""
Convert merged HF model to GGUF for llama.cpp deployment.

Requires llama.cpp repo for convert_hf_to_gguf.py:
  git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
  pip install gguf sentencepiece

Usage:
  python convert_to_gguf.py
"""

import os
import subprocess
import sys

MERGED_PATH = os.path.join(os.path.dirname(__file__), "output", "merged")
OUTPUT_GGUF = os.path.join(os.path.dirname(__file__), "output", "autocomplete-smollm2-135m-q8.gguf")
LLAMA_CPP_DIR = "/tmp/llama.cpp"


def main():
    if not os.path.exists(MERGED_PATH):
        print(f"Error: merged model not found at {MERGED_PATH}")
        print("Run train.py first")
        sys.exit(1)

    if not os.path.exists(LLAMA_CPP_DIR):
        print(f"Cloning llama.cpp to {LLAMA_CPP_DIR}...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggml-org/llama.cpp", LLAMA_CPP_DIR],
            check=True,
        )

    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    f32_gguf = OUTPUT_GGUF.replace("-q8.gguf", "-f32.gguf")

    print("Converting to GGUF F32...")
    subprocess.run(
        [sys.executable, convert_script, MERGED_PATH, "--outfile", f32_gguf, "--outtype", "f32"],
        check=True,
    )

    quantize_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize")
    if os.path.exists(quantize_bin):
        print("Quantizing to Q8_0...")
        subprocess.run([quantize_bin, f32_gguf, OUTPUT_GGUF, "Q8_0"], check=True)
        os.remove(f32_gguf)
    else:
        print(f"llama-quantize not found at {quantize_bin}")
        print(f"F32 GGUF saved to {f32_gguf}")
        print("Quantize manually: llama-quantize {f32_gguf} {OUTPUT_GGUF} Q8_0")
        return

    print(f"\nDone! GGUF model: {OUTPUT_GGUF}")
    size_mb = os.path.getsize(OUTPUT_GGUF) / (1024 * 1024)
    print(f"Size: {size_mb:.0f} MB")
    print(f"\nDeploy: scp {OUTPUT_GGUF} <server>:/root/app/autocomplete/model.gguf")


if __name__ == "__main__":
    main()
