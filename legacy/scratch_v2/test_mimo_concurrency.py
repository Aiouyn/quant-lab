import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(api_key=config.ACTIVE_KEY, base_url=config.ACTIVE_BASE_URL)
print(f"Testing endpoint: {config.ACTIVE_BASE_URL} with model: {config.DEFAULT_MODEL}")

def test_call(idx):
    t0 = time.time()
    try:
        res = client.chat.completions.create(
            model=config.DEFAULT_MODEL,
            messages=[{"role": "user", "content": f"Quant question {idx}: define order flow imbalance in 1 sentence."}],
            max_tokens=60,
            extra_body={"no_think": True},
            timeout=20
        )
        elapsed = time.time() - t0
        return idx, elapsed, res.choices[0].message.content[:80]
    except Exception as e:
        return idx, time.time() - t0, f"ERROR: {e}"

t_start = time.time()
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(test_call, i) for i in range(1, 4)]
    for f in as_completed(futures):
        print(f.result())

print(f"Total concurrency test time: {time.time() - t_start:.2f}s")
