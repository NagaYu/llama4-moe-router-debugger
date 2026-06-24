"""
api.py
======

FastAPI backend for the **Llama 4 MoE Prompt Router & Debugger**.

Responsibilities
----------------
* Lazily load a Hugging Face Llama 4 MoE model (Maverick / Scout) once, behind
  a thread-safe singleton, so the heavy weights are paid for exactly once.
* Run a single *prefill* forward pass per request with :class:`MoERouterCapture`
  armed, capturing ``router_logits`` for every token at every MoE layer.
* Expose the captured :class:`RoutingTrace` as JSON, plus a server-side
  prompt-comparison (diff) endpoint and a low-latency token-streaming endpoint
  (Server-Sent Events) for the dashboard.

Running
-------
::

    # Real model (requires GPU + access to the gated checkpoint):
    export MOE_MODEL_ID="meta-llama/Llama-4-Scout-17B-16E-Instruct"
    uvicorn src.api:app --host 0.0.0.0 --port 8000

    # Weight-free mock mode (laptop / CI / front-end dev):
    export MOE_MOCK=1
    uvicorn src.api:app --reload --port 8000

Environment variables
----------------------
``MOE_MODEL_ID``   HF repo id of the Llama 4 MoE checkpoint.
``MOE_MOCK``       ``1`` to serve deterministic synthetic traces (no weights).
``MOE_DEVICE``     ``auto`` (default), ``cuda``, ``cpu``.
``MOE_DTYPE``      ``bfloat16`` (default), ``float16``, ``float32``.
``MOE_LOAD_4BIT``  ``1`` to load with bitsandbytes 4-bit quantisation.
``MOE_MAX_TOKENS`` Max prompt tokens to instrument (default 256).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .moe_hook import (
    MockRouterCapture,
    MoERouterCapture,
    RoutingTrace,
    diff_traces,
)

logging.basicConfig(
    level=os.environ.get("MOE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("moe_api")


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
class Settings:
    model_id: str = os.environ.get(
        "MOE_MODEL_ID", "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
    mock: bool = os.environ.get("MOE_MOCK", "0") in ("1", "true", "True")
    device: str = os.environ.get("MOE_DEVICE", "auto")
    dtype: str = os.environ.get("MOE_DTYPE", "bfloat16")
    load_4bit: bool = os.environ.get("MOE_LOAD_4BIT", "0") in ("1", "true", "True")
    max_tokens: int = int(os.environ.get("MOE_MAX_TOKENS", "256"))


settings = Settings()


# ──────────────────────────────────────────────────────────────────────────
# Inference engine (thread-safe singleton)
# ──────────────────────────────────────────────────────────────────────────
class InferenceEngine:
    """Owns the model + tokenizer and serialises access to the forward pass.

    A single global lock guards generation because the forward hooks write to a
    shared capture buffer; concurrent forwards would interleave router logits.
    """

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._ready = False
        self._model = None
        self._tokenizer = None
        self._mock: Optional[MockRouterCapture] = None

        if cfg.mock:
            self._mock = MockRouterCapture()
            self._ready = True
            logger.info("InferenceEngine started in MOCK mode (no weights).")

    # ---- loading --------------------------------------------------------
    def ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._load_real_model()
            self._ready = True

    def _load_real_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        logger.info("Loading tokenizer for %s ...", self.cfg.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.cfg.dtype, torch.bfloat16)

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": self.cfg.device if self.cfg.device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }
        if self.cfg.load_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        logger.info("Loading model weights (this can take a while) ...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, **load_kwargs
        )
        self._model.eval()
        logger.info("Model ready in %.1fs.", time.time() - t0)

    # ---- introspection --------------------------------------------------
    def info(self) -> Dict[str, Any]:
        if self.cfg.mock:
            return {
                "mode": "mock",
                "model_id": "mock://synthetic",
                "num_experts": self._mock.num_experts,
                "top_k": self._mock.top_k,
                "num_layers": self._mock.num_layers,
                "ready": True,
            }
        ready = self._ready
        num_experts = top_k = None
        if ready and self._model is not None:
            cfg = getattr(self._model, "config", None)
            tcfg = getattr(cfg, "text_config", cfg)
            num_experts = getattr(tcfg, "num_local_experts", None)
            top_k = getattr(tcfg, "num_experts_per_tok", None)
        return {
            "mode": "model",
            "model_id": self.cfg.model_id,
            "device": self.cfg.device,
            "dtype": self.cfg.dtype,
            "num_experts": num_experts,
            "top_k": top_k,
            "ready": ready,
        }

    # ---- core trace -----------------------------------------------------
    def trace(self, prompt: str) -> RoutingTrace:
        """Run one instrumented prefill pass and return the routing trace."""
        if self.cfg.mock:
            return self._mock.build_trace(prompt)

        self.ensure_loaded()
        import torch

        with self._lock:  # serialise: hooks share a capture buffer
            enc = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.max_tokens,
            )
            input_ids = enc["input_ids"]
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            tokens = [_clean_token(t) for t in tokens]

            device = next(self._model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}

            capture = MoERouterCapture(self._model)
            with capture.session(), torch.no_grad():
                self._model(**enc, use_cache=False)
            return capture.build_trace(prompt, tokens)


def _clean_token(tok: str) -> str:
    """Make BPE tokens human-readable (strip the leading-space marker)."""
    return tok.replace("Ġ", " ").replace("▁", " ").replace("\n", "\\n")


engine = InferenceEngine(settings)


# ──────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ──────────────────────────────────────────────────────────────────────────
class InferRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt to instrument.")


class CompareRequest(BaseModel):
    prompt_a: str = Field(..., min_length=1)
    prompt_b: str = Field(..., min_length=1)


# ──────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Llama 4 MoE Prompt Router & Debugger",
    version="1.0.0",
    description="Capture and visualise per-token expert routing in Llama 4 MoE.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production deployments
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "engine": engine.info()}


@app.get("/config")
def config() -> Dict[str, Any]:
    return engine.info()


@app.post("/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    """Return the full routing trace for a single prompt."""
    try:
        trace = engine.trace(req.prompt)
    except Exception as exc:  # surface model errors cleanly to the UI
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return trace.to_json()


@app.post("/compare")
def compare(req: CompareRequest) -> Dict[str, Any]:
    """Return both traces plus a structural diff for side-by-side comparison."""
    try:
        trace_a = engine.trace(req.prompt_a)
        trace_b = engine.trace(req.prompt_b)
        diff = diff_traces(trace_a, trace_b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Compare failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"trace_a": trace_a.to_json(), "trace_b": trace_b.to_json(), "diff": diff}


@app.get("/infer/stream")
async def infer_stream(prompt: str):
    """Stream per-token routing as Server-Sent Events for a live dashboard.

    The forward pass is computed once (it is inherently non-incremental for the
    prefill), then router rows are emitted token-by-token with a tiny delay so
    the front-end can animate the activation as if it were arriving live.  This
    keeps perceived latency low and the wire format trivial to consume.
    """
    if not prompt:
        raise HTTPException(status_code=400, detail="`prompt` query param required.")

    loop = asyncio.get_event_loop()
    trace = await loop.run_in_executor(None, engine.trace, prompt)
    payload = trace.to_json()

    async def event_gen():
        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "prompt": payload["prompt"],
                    "tokens": payload["tokens"],
                    "num_experts": payload["num_experts"],
                    "top_k": payload["top_k"],
                    "num_layers": len(payload["layers"]),
                }
            ),
        }
        for tok_idx, token in enumerate(payload["tokens"]):
            per_layer = [
                {
                    "layer_index": lyr["layer_index"],
                    "topk_indices": lyr["topk_indices"][tok_idx],
                    "topk_weights": lyr["topk_weights"][tok_idx],
                }
                for lyr in payload["layers"]
                if tok_idx < len(lyr["topk_indices"])
            ]
            yield {
                "event": "token",
                "data": json.dumps(
                    {"index": tok_idx, "token": token, "layers": per_layer}
                ),
            }
            await asyncio.sleep(0.02)  # ~50 tokens/s animation cadence
        yield {"event": "done", "data": json.dumps({"summary": payload["summary"]})}

    return EventSourceResponse(event_gen())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=os.environ.get("MOE_HOST", "0.0.0.0"),
        port=int(os.environ.get("MOE_PORT", "8000")),
        reload=False,
    )
