"""
moe_hook.py
===========

Core instrumentation layer for the **Llama 4 MoE Prompt Router & Debugger**.

This module attaches non-invasive PyTorch *forward hooks* to the router
(gating) sub-modules of a Hugging Face Llama 4 MoE model and captures the
``router_logits`` produced for every token at every MoE layer.  The captured
tensors are converted to plain NumPy / JSON-serialisable structures so they
can be streamed to a web front-end without ever leaking a live ``torch.Tensor``
(which would otherwise pin GPU memory or break JSON encoding).

Design goals
------------
1. **Zero modification of the model forward pass.**  We only register
   ``register_forward_hook`` callbacks; the model graph is never edited and the
   hooks are fully removable (use as a context manager or call ``remove()``).
2. **Architecture-tolerant discovery.**  Llama 4's MoE block is
   ``Llama4TextMoe`` which owns a ``router`` ``nn.Linear`` of shape
   ``(hidden, num_local_experts)``.  Rather than hard-coding the class, we
   discover MoE blocks by structural signature so the tool keeps working across
   minor ``transformers`` refactors and for Scout / Maverick variants alike.
3. **Token-aligned output.**  Captured logits are reshaped to
   ``(num_tokens, num_experts)`` and paired with the decoded tokens so the
   front-end can render a per-token × per-expert heat-map.

The module also ships a :class:`MockRouterCapture` so the dashboard and API can
be developed and demoed without downloading hundreds of gigabytes of weights.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is optional at import time so the mock path works on any machine.
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on torch-less hosts.
    torch = None  # type: ignore
    nn = object  # type: ignore
    _TORCH_AVAILABLE = False


logger = logging.getLogger("moe_hook")


# ──────────────────────────────────────────────────────────────────────────
# Serialisable data structures
# ──────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class LayerRouting:
    """Routing information captured from a single MoE layer.

    Attributes
    ----------
    layer_index:
        Zero-based decoder-block index this MoE layer belongs to.
    num_experts:
        Number of *local* experts the router selects from.
    top_k:
        Number of experts activated per token (``num_experts_per_tok``).
    router_logits:
        ``(num_tokens, num_experts)`` raw gating logits.
    router_probs:
        ``(num_tokens, num_experts)`` **softmax-normalised router preference**.
        This is a visualisation aid showing the router's relative ranking across
        all experts; it is *not* the weight the model applies (see
        ``topk_weights``).  It still drives the per-token heat-map and the
        entropy diagnostic because it is a proper distribution over experts.
    topk_indices:
        ``(num_tokens, top_k)`` indices of the experts actually selected.
    topk_weights:
        ``(num_tokens, top_k)`` **the gate weights Llama 4 actually applies** —
        ``sigmoid`` of the selected logits, exactly mirroring
        ``modeling_llama4.Llama4Router`` (which scatters ``-inf`` onto the
        unselected experts and then takes the sigmoid).  Each value lies in
        ``(0, 1)`` and they are deliberately *not* re-normalised, because
        Llama 4 does not re-normalise them either.
    """

    layer_index: int
    num_experts: int
    top_k: int
    router_logits: np.ndarray
    router_probs: np.ndarray
    topk_indices: np.ndarray
    topk_weights: np.ndarray

    def to_json(self) -> Dict[str, Any]:
        """Convert to a JSON-friendly dict (lists, not ndarrays)."""
        return {
            "layer_index": int(self.layer_index),
            "num_experts": int(self.num_experts),
            "top_k": int(self.top_k),
            "router_logits": _round(self.router_logits),
            "router_probs": _round(self.router_probs),
            "topk_indices": self.topk_indices.astype(int).tolist(),
            "topk_weights": _round(self.topk_weights),
        }


@dataclasses.dataclass
class RoutingTrace:
    """A complete routing trace for one prompt.

    This is the canonical structure transferred from the API to the UI and the
    unit on which the *diff* (prompt-comparison) feature operates.
    """

    prompt: str
    tokens: List[str]
    num_experts: int
    top_k: int
    layers: List[LayerRouting] = dataclasses.field(default_factory=list)

    # ---- aggregate views ------------------------------------------------
    def expert_usage_matrix(self) -> np.ndarray:
        """Mean activation probability per ``(layer, expert)``.

        Returns ``(num_layers, num_experts)`` averaged over tokens — the
        signal that drives the "which experts light up" overview heat-map.
        """
        if not self.layers:
            return np.zeros((0, self.num_experts), dtype=np.float32)
        return np.stack([lyr.router_probs.mean(axis=0) for lyr in self.layers])

    def expert_load(self) -> np.ndarray:
        """Fraction of (token, slot) selections each expert received overall.

        Returns ``(num_experts,)`` — a load-balancing fingerprint.  A perfectly
        balanced MoE would be uniform at ``top_k / num_experts``.
        """
        counts = np.zeros(self.num_experts, dtype=np.float64)
        total = 0
        for lyr in self.layers:
            idx, cnt = np.unique(lyr.topk_indices, return_counts=True)
            counts[idx] += cnt
            total += lyr.topk_indices.size
        return (counts / total).astype(np.float32) if total else counts.astype(np.float32)

    def routing_entropy(self) -> List[float]:
        """Per-layer mean routing entropy (in nats).

        High entropy ⇒ the router is "undecided" / spreading mass across many
        experts; low entropy ⇒ confident, specialised routing.  This is one of
        the most useful debugging signals for prompt engineering.
        """
        out: List[float] = []
        for lyr in self.layers:
            p = np.clip(lyr.router_probs, 1e-12, 1.0)
            ent = -(p * np.log(p)).sum(axis=1)  # per token
            out.append(float(ent.mean()))
        return out

    def to_json(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "tokens": self.tokens,
            "num_experts": int(self.num_experts),
            "top_k": int(self.top_k),
            "layers": [lyr.to_json() for lyr in self.layers],
            "summary": {
                "expert_usage": _round(self.expert_usage_matrix()),
                "expert_load": _round(self.expert_load()),
                "routing_entropy": [round(x, 6) for x in self.routing_entropy()],
            },
        }


def _round(arr: np.ndarray, ndigits: int = 6) -> List[Any]:
    """Round + listify an ndarray for compact, deterministic JSON."""
    return np.round(np.asarray(arr, dtype=np.float64), ndigits).tolist()


# ──────────────────────────────────────────────────────────────────────────
# The hook engine
# ──────────────────────────────────────────────────────────────────────────
class MoERouterCapture:
    """Register removable forward hooks on a Llama 4 MoE model's routers.

    Typical usage::

        capture = MoERouterCapture(model)
        with capture.session():            # arms the hooks
            model.generate(**inputs)
        trace = capture.build_trace(prompt, tokens)

    The class is *re-entrant per generation step*: each forward call appends a
    fresh batch of logits, so we keep only the **prefill** pass (the one that
    contains every prompt token) by default — that is what the dashboard needs.
    Set ``keep_all_steps=True`` to retain decode steps as well.
    """

    #: Module-class name fragments that identify an MoE block.
    MOE_CLASS_HINTS: Tuple[str, ...] = ("Moe", "MoE", "SparseMoe")
    #: Attribute names commonly used for the gating Linear inside an MoE block.
    ROUTER_ATTR_HINTS: Tuple[str, ...] = ("router", "gate")

    def __init__(
        self,
        model: "nn.Module",
        *,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
        keep_all_steps: bool = False,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not available. Use MockRouterCapture for a "
                "weight-free demo, or install torch + transformers."
            )
        self.model = model
        self.keep_all_steps = keep_all_steps

        cfg = getattr(model, "config", None)
        # Llama 4 nests the text config under `.text_config`.
        text_cfg = getattr(cfg, "text_config", cfg)
        self.num_experts = int(
            num_experts
            or getattr(text_cfg, "num_local_experts", 0)
            or getattr(text_cfg, "num_experts", 0)
            or 0
        )
        self.top_k = int(
            top_k
            or getattr(text_cfg, "num_experts_per_tok", 0)
            or getattr(text_cfg, "num_experts_per_token", 0)
            or 1
        )

        self._handles: List[Any] = []
        # layer_index -> list of (num_tokens, num_experts) arrays, one per step
        self._buffer: Dict[int, List[np.ndarray]] = {}
        self._router_modules = self._discover_routers()
        if not self._router_modules:
            logger.warning(
                "No MoE router modules were discovered. The model may be dense "
                "or use an unrecognised architecture."
            )

    # ---- discovery ------------------------------------------------------
    def _discover_routers(self) -> List[Tuple[int, "nn.Module"]]:
        """Locate the gating Linear of every MoE block in decoder order."""
        found: List[Tuple[int, "nn.Module"]] = []
        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__
            if not any(h in cls_name for h in self.MOE_CLASS_HINTS):
                continue
            router = self._extract_router(module)
            if router is None:
                continue
            layer_index = self._parse_layer_index(name)
            found.append((layer_index, router))
        # Stable ordering by decoder-block index.
        found.sort(key=lambda t: t[0])
        logger.info("Discovered %d MoE router modules.", len(found))
        if not found:
            logger.warning(
                "No routers discovered. If you enabled the Transformers "
                "`kernels` integration (use_kernel_forward_from_hub), the fused "
                "MoE kernel can bypass the Python `router` submodule and the "
                "hook will not fire — run with the default eager forward to "
                "instrument routing."
            )
        return found

    def _extract_router(self, moe_module: "nn.Module") -> Optional["nn.Module"]:
        for attr in self.ROUTER_ATTR_HINTS:
            cand = getattr(moe_module, attr, None)
            if isinstance(cand, nn.Module):
                return cand
        # Fallback: a direct child Linear whose out_features == num_experts.
        for child in moe_module.children():
            if isinstance(child, nn.Linear) and (
                self.num_experts == 0 or child.out_features == self.num_experts
            ):
                return child
        return None

    @staticmethod
    def _parse_layer_index(module_name: str) -> int:
        m = re.search(r"layers\.(\d+)", module_name)
        return int(m.group(1)) if m else -1

    # ---- hook callback --------------------------------------------------
    def _make_hook(self, layer_index: int):
        def _hook(_module, _inputs, output):
            logits = self._coerce_logits(output)
            if logits is None:
                return
            arr = logits.detach().to(torch.float32).cpu().numpy()
            # Collapse any leading (batch, seq) dims to a flat token axis.
            arr = arr.reshape(-1, arr.shape[-1])
            self._buffer.setdefault(layer_index, []).append(arr)

        return _hook

    @staticmethod
    def _coerce_logits(output: Any) -> Optional["torch.Tensor"]:
        """Normalise whatever the router/MoE module returned to a logits tensor.

        Handles three shapes seen in the wild:
        * a plain Linear output tensor (router-Linear hook),
        * a tuple whose router_logits element is the last/2nd item (MoE hook),
        * a tensor that is already ``(*, num_experts)``.
        """
        if torch is None:
            return None
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            # Prefer a 2-D tensor; router_logits is typically the trailing item.
            for item in reversed(output):
                if isinstance(item, torch.Tensor) and item.dim() >= 2:
                    return item
        return None

    # ---- lifecycle ------------------------------------------------------
    def arm(self) -> "MoERouterCapture":
        """Attach the forward hooks. Idempotent."""
        if self._handles:
            return self
        for layer_index, router in self._router_modules:
            handle = router.register_forward_hook(self._make_hook(layer_index))
            self._handles.append(handle)
        logger.debug("Armed %d router hooks.", len(self._handles))
        return self

    def remove(self) -> None:
        """Detach all hooks and clear handles (buffer is preserved)."""
        for h in self._handles:
            with contextlib.suppress(Exception):
                h.remove()
        self._handles.clear()

    def reset(self) -> None:
        """Clear captured data between prompts."""
        self._buffer.clear()

    @contextlib.contextmanager
    def session(self):
        """Context manager that arms hooks for the duration of a forward pass."""
        self.reset()
        self.arm()
        try:
            yield self
        finally:
            self.remove()

    # ---- result assembly ------------------------------------------------
    def build_trace(self, prompt: str, tokens: Sequence[str]) -> RoutingTrace:
        """Fold the raw buffer into a :class:`RoutingTrace`.

        Only the first forward pass per layer (the prefill, which holds all
        prompt tokens) is used unless ``keep_all_steps`` was set, in which case
        decode steps are concatenated along the token axis.
        """
        layers: List[LayerRouting] = []
        n_tokens = len(tokens)
        for layer_index in sorted(self._buffer):
            steps = self._buffer[layer_index]
            if not steps:
                continue
            logits = np.concatenate(steps, axis=0) if self.keep_all_steps else steps[0]
            # Align to the number of prompt tokens when possible (prefill).
            if not self.keep_all_steps and logits.shape[0] >= n_tokens > 0:
                logits = logits[:n_tokens]
            layers.append(self._finalise_layer(layer_index, logits))

        return RoutingTrace(
            prompt=prompt,
            tokens=list(tokens),
            num_experts=self.num_experts or (layers[0].num_experts if layers else 0),
            top_k=self.top_k,
            layers=layers,
        )

    def _finalise_layer(self, layer_index: int, logits: np.ndarray) -> LayerRouting:
        num_experts = logits.shape[-1]
        # softmax(logits): a normalised *preference* distribution over all
        # experts — used for the heat-map and entropy. NOT the applied gate.
        probs = _softmax(logits, axis=-1)
        k = min(self.top_k, num_experts)
        topk_idx = np.argsort(-logits, axis=-1)[:, :k]
        topk_logits = np.take_along_axis(logits, topk_idx, axis=-1)
        # The gate Llama 4 ACTUALLY applies: sigmoid of the selected logits.
        # Llama4Router scatters -inf onto unselected experts then takes sigmoid,
        # so unselected weights collapse to 0 and selected ones are sigmoid(z),
        # never re-normalised. We reproduce that here for the top-k slots.
        topk_w = _sigmoid(topk_logits)
        return LayerRouting(
            layer_index=layer_index,
            num_experts=num_experts,
            top_k=k,
            router_logits=logits.astype(np.float32),
            router_probs=probs.astype(np.float32),
            topk_indices=topk_idx.astype(np.int32),
            topk_weights=topk_w.astype(np.float32),
        )


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic sigmoid, matching the gate Llama 4 applies.
    # Evaluate each branch only on its own mask so the large-magnitude inputs
    # never feed a np.exp that would overflow (np.where evaluates both sides).
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Prompt comparison (diff)
# ──────────────────────────────────────────────────────────────────────────
def diff_traces(a: RoutingTrace, b: RoutingTrace) -> Dict[str, Any]:
    """Compute a structural diff between two routing traces.

    The diff is expressed on the *aggregate* expert-usage matrix so that two
    prompts of different lengths remain comparable.  Positive values in
    ``usage_delta`` mean prompt **B** activates that ``(layer, expert)`` more
    than prompt **A``.

    Returns a JSON-serialisable dict consumed directly by the dashboard's
    "Compare" tab.
    """
    if a.num_experts != b.num_experts:
        raise ValueError("Traces have different expert counts; not comparable.")

    ua = a.expert_usage_matrix()
    ub = b.expert_usage_matrix()
    n_layers = min(ua.shape[0], ub.shape[0])
    ua, ub = ua[:n_layers], ub[:n_layers]
    usage_delta = ub - ua

    load_a, load_b = a.expert_load(), b.expert_load()
    ent_a, ent_b = a.routing_entropy(), b.routing_entropy()
    m = min(len(ent_a), len(ent_b))

    return {
        "num_experts": int(a.num_experts),
        "num_layers": int(n_layers),
        "usage_a": _round(ua),
        "usage_b": _round(ub),
        "usage_delta": _round(usage_delta),
        "load_a": _round(load_a),
        "load_b": _round(load_b),
        "load_delta": _round(load_b - load_a),
        "entropy_a": [round(x, 6) for x in ent_a[:m]],
        "entropy_b": [round(x, 6) for x in ent_b[:m]],
        # L1 distance per layer: a single scalar "how differently did the
        # router behave" score, handy for sorting/triage.
        "divergence_per_layer": _round(np.abs(usage_delta).sum(axis=1)),
        "total_divergence": float(np.abs(usage_delta).sum()),
    }


# ──────────────────────────────────────────────────────────────────────────
# Mock backend (weight-free development / CI)
# ──────────────────────────────────────────────────────────────────────────
class MockRouterCapture:
    """Generate plausible, deterministic routing traces without any model.

    Useful for front-end work, demos, and CI where downloading Llama 4 weights
    is impractical.  The synthetic router is *prompt-conditioned* (seeded by a
    hash of the prompt) so that two different prompts produce meaningfully
    different — and reproducible — activation patterns, exercising the diff
    feature end to end.
    """

    def __init__(self, num_experts: int = 16, top_k: int = 1, num_layers: int = 24):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_layers = num_layers

    @staticmethod
    def tokenize(prompt: str) -> List[str]:
        toks = re.findall(r"\w+|[^\w\s]", prompt, flags=re.UNICODE)
        return toks or ["<empty>"]

    def build_trace(self, prompt: str, tokens: Optional[Sequence[str]] = None) -> RoutingTrace:
        tokens = list(tokens) if tokens is not None else self.tokenize(prompt)
        # Seed from a stable hash (sha256) rather than the builtin hash(), which
        # is salted per-process — this keeps traces reproducible across restarts.
        seed = int.from_bytes(
            hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "little"
        )
        rng = np.random.default_rng(seed)
        layers: List[LayerRouting] = []
        n = len(tokens)
        for layer_index in range(self.num_layers):
            # Give each layer a "preferred" set of experts so the heat-map has
            # structure rather than noise, and let token identity tilt it.
            bias = rng.normal(0, 1.5, size=self.num_experts)
            tok_tilt = rng.normal(0, 0.8, size=(n, self.num_experts))
            logits = bias[None, :] + tok_tilt + rng.normal(0, 0.5, size=(n, self.num_experts))
            layers.append(
                # Reuse the exact production math (softmax / top-k / renorm).
                MoERouterCapture._finalise_layer(
                    _DummyK(self.top_k), layer_index, logits.astype(np.float32)
                )
            )
        return RoutingTrace(
            prompt=prompt,
            tokens=tokens,
            num_experts=self.num_experts,
            top_k=self.top_k,
            layers=layers,
        )


class _DummyK:
    """Minimal shim so MockRouterCapture can reuse ``_finalise_layer``."""

    def __init__(self, top_k: int):
        self.top_k = top_k


__all__ = [
    "LayerRouting",
    "RoutingTrace",
    "MoERouterCapture",
    "MockRouterCapture",
    "diff_traces",
]
