"""Generate a chat-style token corpus from the model itself (default sampling),
for ranking a hot-token draft vocabulary. Writes chat_corpus_ids.json."""
import concurrent.futures as cf
import itertools
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import BASE, MODEL  # noqa: E402

TOPICS = ["the history of the Roman Empire", "how vaccines work", "home network security", "sourdough baking",
          "personal finance and budgeting", "climate change", "learning a new language", "FPV drones",
          "electric cars vs hybrids", "how LLMs are trained", "gardening in a cold climate", "the stock market",
          "writing a cover letter", "planning a trip to Japan", "sleep and health", "the French Revolution",
          "renting vs buying a home", "black holes", "basic car maintenance", "raising a puppy",
          "Linux system administration", "email etiquette at work", "meal prep for a week", "quantum computing",
          "a Python script that renames files by date"]
TASKS = ["Explain {t} to a beginner.", "Give me a detailed overview of {t}.", "What are common misconceptions about {t}?",
         "Write a short blog post about {t}.", "Summarize the key points of {t} as a bulleted list.",
         "I'm confused about {t}. Can you help me understand it step by step?", "Compare two different approaches to {t}.",
         "Write an email to a friend recommending they learn about {t}.", "What should I know before getting into {t}?",
         "Create a study plan for {t}.", "Tell me a story that involves {t}.", "Answer as an expert: what's the hardest part of {t}?"]


def gen(prompt: str) -> list[int]:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 700,
            "logprobs": True, "chat_template_kwargs": {"enable_thinking": random.random() < 0.3}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return [t["token"] for t in r["choices"][0]["logprobs"]["content"]]


random.seed(0)
prompts = [task.format(t=topic) for topic, task in itertools.product(TOPICS, TASKS)]
random.shuffle(prompts)
prompts = prompts[: int(sys.argv[1]) if len(sys.argv) > 1 else 200]
with cf.ThreadPoolExecutor(4) as ex:
    token_strs = [t for toks in ex.map(gen, prompts) for t in toks]
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("/opt/d/models/Qwen3.8-Flash-Next-NVFP4")
# logprob tokens are decoded strings; re-tokenize the concatenated text
ids = tok("".join(token_strs), add_special_tokens=False)["input_ids"]
json.dump(ids, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_corpus_ids.json"), "w"))
print(f"{len(prompts)} prompts -> {len(ids)} tokens")
