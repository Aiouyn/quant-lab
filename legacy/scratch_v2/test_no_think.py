import sys, time
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import OpenAI
import config

client = OpenAI(api_key=config.ACTIVE_KEY, base_url=config.ACTIVE_BASE_URL)

tests = [
    ("prefix /no_think", {"messages": [{"role": "user", "content": "/no_think Return valid JSON with answer: 42"}]}),
    ("extra_body no_think", {"extra_body": {"no_think": True}}),
    ("extra_body thinking type disabled", {"extra_body": {"thinking": {"type": "disabled"}}}),
    ("reasoning_effort low", {"reasoning_effort": "low"})
]

for name, kwargs in tests:
    print(f"\n--- Testing {name} ---")
    t0 = time.time()
    try:
        msgs = kwargs.pop("messages", [{"role": "user", "content": "Return valid JSON with answer: 42"}])
        resp = client.chat.completions.create(
            model="mimo-v2.6-pro",
            messages=msgs,
            max_tokens=300,
            timeout=20,
            **kwargs
        )
        msg = resp.choices[0].message
        print(f"Done in {time.time()-t0:.2f}s! Finish: {resp.choices[0].finish_reason}")
        print("Content:", msg.content)
        print("Reasoning length:", len(getattr(msg, "reasoning_content", "") or ""))
    except Exception as e:
        print("Error:", e)
