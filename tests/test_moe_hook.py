"""
Smoke + correctness tests for the MoE hook core.

These run **without torch or any model weights** by driving the deterministic
:class:`MockRouterCapture`, which reuses the exact production math in
``MoERouterCapture._finalise_layer``. So a green test suite verifies the parts
of the pipeline that are model-independent: shape contracts, the Llama 4 gating
math (softmax preference + sigmoid applied gate), JSON serialisability,
determinism, the aggregate diagnostics, and the prompt-comparison diff.

Run:  pytest -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.moe_hook import (
    MockRouterCapture,
    RoutingTrace,
    _sigmoid,
    _softmax,
    diff_traces,
)

NUM_EXPERTS = 16
TOP_K = 1
NUM_LAYERS = 6

PROMPT_A = "Write a Python function to sort a list."
PROMPT_B = "Write a haiku about the ocean."


@pytest.fixture()
def mock() -> MockRouterCapture:
    return MockRouterCapture(
        num_experts=NUM_EXPERTS, top_k=TOP_K, num_layers=NUM_LAYERS
    )


@pytest.fixture()
def trace_a(mock: MockRouterCapture) -> RoutingTrace:
    return mock.build_trace(PROMPT_A)


# ── math helpers ──────────────────────────────────────────────────────────
def test_softmax_is_a_distribution():
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    p = _softmax(x, axis=-1)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-6)
    assert (p >= 0).all()


def test_sigmoid_matches_reference_and_is_stable():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    got = _sigmoid(x)
    ref = 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))
    np.testing.assert_allclose(got, ref, atol=1e-9)
    assert np.isfinite(got).all()          # no overflow at the extremes
    assert (0.0 <= got).all() and (got <= 1.0).all()


# ── trace shape contracts ─────────────────────────────────────────────────
def test_trace_shapes(trace_a: RoutingTrace):
    n_tok = len(trace_a.tokens)
    assert trace_a.num_experts == NUM_EXPERTS
    assert len(trace_a.layers) == NUM_LAYERS
    for lyr in trace_a.layers:
        assert lyr.router_logits.shape == (n_tok, NUM_EXPERTS)
        assert lyr.router_probs.shape == (n_tok, NUM_EXPERTS)
        assert lyr.topk_indices.shape == (n_tok, TOP_K)
        assert lyr.topk_weights.shape == (n_tok, TOP_K)


def test_router_probs_sum_to_one(trace_a: RoutingTrace):
    for lyr in trace_a.layers:
        np.testing.assert_allclose(lyr.router_probs.sum(axis=-1), 1.0, rtol=1e-5)


def test_topk_weights_are_sigmoid_gates(trace_a: RoutingTrace):
    """topk_weights must equal sigmoid(selected logits) — Llama 4's applied gate,
    NOT a renormalised softmax (which would collapse to 1.0 for top_k=1)."""
    for lyr in trace_a.layers:
        sel_logits = np.take_along_axis(
            lyr.router_logits, lyr.topk_indices, axis=-1
        )
        np.testing.assert_allclose(lyr.topk_weights, _sigmoid(sel_logits), atol=1e-6)
        assert (lyr.topk_weights > 0).all() and (lyr.topk_weights < 1).all()
        # Regression guard: must not be the degenerate all-ones from renorm.
        assert not np.allclose(lyr.topk_weights, 1.0)


def test_topk_indices_are_argmax(trace_a: RoutingTrace):
    for lyr in trace_a.layers:
        true_top = np.argmax(lyr.router_logits, axis=-1)
        np.testing.assert_array_equal(lyr.topk_indices[:, 0], true_top)


# ── aggregate diagnostics ─────────────────────────────────────────────────
def test_expert_usage_matrix_shape(trace_a: RoutingTrace):
    usage = trace_a.expert_usage_matrix()
    assert usage.shape == (NUM_LAYERS, NUM_EXPERTS)
    assert (usage >= 0).all()


def test_expert_load_is_a_distribution(trace_a: RoutingTrace):
    load = trace_a.expert_load()
    assert load.shape == (NUM_EXPERTS,)
    np.testing.assert_allclose(load.sum(), 1.0, rtol=1e-6)


def test_routing_entropy_bounds(trace_a: RoutingTrace):
    ent = trace_a.routing_entropy()
    assert len(ent) == NUM_LAYERS
    max_ent = np.log(NUM_EXPERTS)  # uniform distribution upper bound
    for e in ent:
        assert 0.0 <= e <= max_ent + 1e-6


# ── serialisation & determinism ───────────────────────────────────────────
def test_to_json_is_serialisable(trace_a: RoutingTrace):
    payload = trace_a.to_json()
    # Must round-trip through JSON with no ndarray / numpy scalar leakage.
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["num_experts"] == NUM_EXPERTS
    assert len(decoded["tokens"]) == len(trace_a.tokens)
    assert "summary" in decoded
    assert set(decoded["summary"]) == {
        "expert_usage",
        "expert_load",
        "routing_entropy",
    }


def test_determinism(mock: MockRouterCapture):
    a1 = mock.build_trace(PROMPT_A).to_json()
    a2 = mock.build_trace(PROMPT_A).to_json()
    assert a1 == a2


def test_different_prompts_diverge(mock: MockRouterCapture):
    a = mock.build_trace(PROMPT_A).to_json()
    b = mock.build_trace(PROMPT_B).to_json()
    assert a["summary"]["expert_usage"] != b["summary"]["expert_usage"]


# ── prompt comparison (diff) ──────────────────────────────────────────────
def test_diff_structure_and_antisymmetry(mock: MockRouterCapture):
    a = mock.build_trace(PROMPT_A)
    b = mock.build_trace(PROMPT_B)
    d_ab = diff_traces(a, b)
    d_ba = diff_traces(b, a)

    assert d_ab["num_experts"] == NUM_EXPERTS
    assert d_ab["num_layers"] == NUM_LAYERS
    assert np.array(d_ab["usage_delta"]).shape == (NUM_LAYERS, NUM_EXPERTS)

    # delta(B-A) == -delta(A-B); total divergence is symmetric.
    np.testing.assert_allclose(
        d_ab["usage_delta"], -np.array(d_ba["usage_delta"]), atol=1e-6
    )
    assert d_ab["total_divergence"] == pytest.approx(d_ba["total_divergence"])
    assert d_ab["total_divergence"] > 0


def test_diff_of_identical_traces_is_zero(mock: MockRouterCapture):
    a = mock.build_trace(PROMPT_A)
    d = diff_traces(a, a)
    assert d["total_divergence"] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(d["usage_delta"], 0.0)


def test_diff_rejects_incompatible_expert_counts():
    a = MockRouterCapture(num_experts=16, top_k=1, num_layers=4).build_trace("x")
    b = MockRouterCapture(num_experts=8, top_k=1, num_layers=4).build_trace("y")
    with pytest.raises(ValueError):
        diff_traces(a, b)
