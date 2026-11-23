"""
Shared FREE-model LLM client for the agentic-robotics experiments.

- Routes across multiple FREE OpenAI-compatible backends (keys read from the
  local opencode auth store -- the same free models opencode exposes):
    1. NVIDIA NIM  (https://integrate.api.nvidia.com/v1)  -- primary
    2. OpenRouter free models                              -- fallback
  On rate-limit / empty reply it rotates to the next backend.
- On-disk cache keyed by (primary-model, messages, temperature, seed) for
  reproducibility and to avoid re-paying latency on repeated runs.
- Full JSONL call log (model, latency, cache flag) for auditability.
- Circuit breaker: after repeated live failures (free tier throttled) it
  short-circuits so callers fall back to their deterministic path immediately
  instead of hanging.

No API keys are hardcoded. Nothing here needs a GPU or dataset.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

import httpx

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# FREE models verified to return clean output (2026-08).
# Open-weight models verified on the free tiers (2026-11). The OpenRouter entry is
# the same model, used only as a fallback when NIM rate-limits.
NVIDIA_MODELS = ["nvidia/nemotron-3.5-lightning-30b-a3b"]
OPENROUTER_FREE_MODELS = ["nvidia/nemotron-3.5-lightning:free"]
OLLAMA_URL = "http://localhost:11434/api/chat"  # native API (lets us disable thinking)

_CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", Path(__file__).parent.parent / "results" / "_llm_cache"))  # gitignored
_LOG_DIR = Path(os.environ.get("LLM_LOG_DIR", Path(__file__).parent.parent / "results" / "_llm_logs"))


def _auth() -> dict:
    p = Path.home() / ".local/share/opencode/auth.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def default_backends() -> List[Dict]:
    """Ordered list of free OpenAI-compatible backends that are available."""
    if os.environ.get("SEL_BACKEND") == "nim":
        return [{"url": NVIDIA_URL, "key": os.environ.get("NVIDIA_API_KEY") or (_auth().get("nvidia") or {}).get("key"),
                 "model": NVIDIA_MODELS[0], "kind": "nvidia"}]
    if os.environ.get("SEL_BACKEND") == "ollama":
        return [{"url": OLLAMA_URL, "key": "ollama", "kind": "ollama",
                 "model": os.environ.get("OLLAMA_MODEL", "qwen3:14b")}]
    a = _auth()
    backends: List[Dict] = []
    nv = os.environ.get("NVIDIA_API_KEY") or (a.get("nvidia") or {}).get("key")
    if nv:
        for m in NVIDIA_MODELS:
            backends.append({"url": NVIDIA_URL, "key": nv, "model": m, "kind": "nvidia"})
    ork = os.environ.get("OPENROUTER_API_KEY") or (a.get("openrouter") or {}).get("key")
    if ork:
        for m in OPENROUTER_FREE_MODELS:
            backends.append({"url": OPENROUTER_URL, "key": ork, "model": m, "kind": "openrouter"})
    return backends


import threading
_PACE_LOCK = threading.Lock()
_LAST_CALL = [0.0]


def _global_pace(pace: float):
    """Process-wide minimum spacing between live calls (all threads share one rate limit)."""
    if pace <= 0:
        return
    with _PACE_LOCK:
        dt = pace - (time.time() - _LAST_CALL[0])
        if dt > 0:
            time.sleep(dt)
        _LAST_CALL[0] = time.time()


class FreeLLM:
    def __init__(self, backends: Optional[List[Dict]] = None,
                 models: Optional[List[str]] = None,  # accepted for back-compat; ignored
                 temperature: float = 0.2, max_tokens: int = 1600,
                 log_name: str = "calls", seed: Optional[int] = None):
        self.backends = backends or default_backends()
        if not self.backends:
            raise RuntimeError("No free LLM backends available (no NVIDIA/OpenRouter key).")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = _LOG_DIR / f"{log_name}.jsonl"
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model_counts: Dict[str, int] = {}
        self.n_cache_hits = 0
        self._consec_fail = 0
        self._breaker_trip = int(os.environ.get("LLM_BREAKER_TRIP", "8"))
        self._breaker_until = 0.0
        self._breaker_cooldown = float(os.environ.get("LLM_BREAKER_COOLDOWN", "90"))
        self.n_live_fail = 0
        self.n_breaker_skips = 0
        # pacing + persistence for grounded runs on rate-limited free tiers.
        self.pace = float(os.environ.get("LLM_PACE", "0"))          # min secs between live calls
        self.persist = os.environ.get("LLM_PERSIST", "0") == "1"    # retry hard, never give up early
        self._last_call = 0.0

    @property
    def primary_model(self) -> str:
        return self.backends[0]["model"]

    def _cache_key(self, messages, temperature) -> str:
        return hashlib.sha256(json.dumps(
            {"m": self.primary_model, "msgs": messages, "t": temperature, "s": self.seed},
            sort_keys=True).encode()).hexdigest()

    def _log(self, rec: Dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def chat(self, messages: List[Dict], temperature: Optional[float] = None,
             max_tokens: Optional[int] = None, use_cache: bool = True) -> str:
        temp = self.temperature if temperature is None else temperature
        mt = self.max_tokens if max_tokens is None else max_tokens

        ck = self._cache_key(messages, temp)
        cache_file = _CACHE_DIR / f"{ck}.json"
        if use_cache and cache_file.exists():
            self.n_cache_hits += 1
            rec = json.loads(cache_file.read_text())
            self.model_counts[rec.get("model")] = self.model_counts.get(rec.get("model"), 0) + 1
            self._log({"ts": time.time(), "cache": True, "model": rec.get("model"),
                       "reply_len": len(rec.get("reply", ""))})
            return rec["reply"]

        if not self.persist and time.time() < self._breaker_until:
            self.n_breaker_skips += 1
            raise RuntimeError("LLM circuit breaker open (free tier throttled)")

        # In persist mode we cycle the backends many times with capped backoff so
        # every unique decision-state eventually gets a real free-model answer
        # (grounded + cached) rather than falling back. Paced to respect free-tier
        # rate limits.
        rounds = 8 if self.persist else 1
        last_err = None
        for _round in range(rounds):
          for be in self.backends:
            for attempt in range(2):
                _global_pace(self.pace)
                t0 = time.time()
                try:
                    if be["kind"] == "ollama":
                        payload = {"model": be["model"], "messages": messages, "stream": False,
                                   "think": False, "options": {"temperature": temp, "num_predict": min(mt, 400),
                                                               "num_ctx": int(os.environ.get("OLLAMA_CTX", "8192")),
                                                               "seed": 0}}
                    else:
                        payload = {"model": be["model"], "messages": messages,
                                   "temperature": temp, "max_tokens": mt}
                    if self.seed is not None and be["kind"] != "ollama":
                        payload["seed"] = self.seed
                    with httpx.Client(timeout=httpx.Timeout(float(os.environ.get("LLM_TIMEOUT", "90")), connect=10.0)) as c:
                        r = c.post(be["url"],
                                   headers={"Authorization": f"Bearer {be['key']}",
                                            "Content-Type": "application/json"},
                                   json=payload)
                    if r.status_code == 429:
                        last_err = f"429 {be['kind']}/{be['model']}"
                        time.sleep(min(2 ** attempt * 5, 20))
                        continue
                    r.raise_for_status()
                    data = r.json()
                    if be["kind"] == "ollama":
                        msg = data.get("message", {})
                        self.prompt_tokens += data.get("prompt_eval_count") or 0
                        self.completion_tokens += data.get("eval_count") or 0
                    else:
                        msg = (data.get("choices") or [{}])[0].get("message", {})
                        u = data.get("usage") or {}
                        self.prompt_tokens += u.get("prompt_tokens") or 0
                        self.completion_tokens += u.get("completion_tokens") or 0
                    # only the final answer; never parse the reasoning trace (it contains
                    # example actions copied from the prompt)
                    reply = (msg.get("content") or "").strip()
                    if not reply:
                        last_err = f"empty {be['kind']}/{be['model']}"
                        break
                    self.n_calls += 1
                    self.model_counts[be["model"]] = self.model_counts.get(be["model"], 0) + 1
                    self._consec_fail = 0
                    self._log({"ts": time.time(), "cache": False, "kind": be["kind"],
                               "model": be["model"], "latency_s": round(time.time() - t0, 2),
                               "reply_len": len(reply)})
                    if use_cache:
                        cache_file.write_text(json.dumps({"model": be["model"], "reply": reply}))
                    return reply
                except Exception as e:
                    last_err = f"{type(e).__name__}:{str(e)[:80]} ({be['kind']}/{be['model']})"
                    time.sleep(min(2 ** attempt * 2, 6))
        self._consec_fail += 1
        self.n_live_fail += 1
        if self._consec_fail >= self._breaker_trip:
            self._breaker_until = time.time() + self._breaker_cooldown
        raise RuntimeError(f"All free backends failed. Last error: {last_err}")


def extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from an LLM reply (handles code fences / prose and
    bare unquoted identifiers such as {"target":table})."""
    import re
    t = re.sub(r'(:\s*)([A-Za-z_][A-Za-z_ ]*?)(\s*[,}])', lambda m: m.group(1) + json.dumps(m.group(2))
               + m.group(3) if m.group(2) not in ("true", "false", "null") else m.group(0), text.strip())
    if t.startswith("```"):
        nl = t.find("\n")
        if nl >= 0:
            t = t[nl + 1:]
        t = t.split("```")[0].strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    end = t.rfind("}") + 1
    if 0 <= start < end:
        try:
            return json.loads(t[start:end])
        except Exception:
            return None
    return None


if __name__ == "__main__":
    llm = FreeLLM(log_name="smoke", temperature=0.1)
    print("backends:", [(b["kind"], b["model"]) for b in llm.backends])
    out = llm.chat([{"role": "user", "content": 'Return only JSON: {"ok": true, "n": 3}'}])
    print("reply:", out)
    print("parsed:", extract_json(out))
    print("calls:", llm.n_calls, "cache_hits:", llm.n_cache_hits, "live_fail:", llm.n_live_fail)
