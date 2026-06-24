"""
app.py
======

Streamlit dashboard for the **Llama 4 MoE Prompt Router & Debugger**.

It talks to the FastAPI backend (:mod:`src.api`) and renders three views:

1. **Inspect** — per-token × per-expert routing heat-map for one prompt, plus
   load-balancing and routing-entropy diagnostics.
2. **Compare** — side-by-side activation of two prompts with a signed *diff*
   heat-map so you can see exactly which experts a prompt change re-routes.
3. **Per-token drill-down** — pick a token, see which experts each layer fired.

Run::

    export MOE_API_URL="http://localhost:8000"   # optional, this is the default
    streamlit run src/app.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("MOE_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.environ.get("MOE_API_TIMEOUT", "600"))

st.set_page_config(
    page_title="Llama 4 MoE Router & Debugger",
    page_icon="🧭",
    layout="wide",
)


# ──────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=30)
def fetch_config() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_URL}/config", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_infer(prompt: str) -> Dict[str, Any]:
    r = requests.post(f"{API_URL}/infer", json={"prompt": prompt}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_compare(prompt_a: str, prompt_b: str) -> Dict[str, Any]:
    r = requests.post(
        f"{API_URL}/compare",
        json={"prompt_a": prompt_a, "prompt_b": prompt_b},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────────────────
# Plot builders
# ──────────────────────────────────────────────────────────────────────────
def heatmap(
    z: np.ndarray,
    *,
    x_title: str,
    y_title: str,
    colorscale: str = "Viridis",
    zmid: Optional[float] = None,
    title: str = "",
    x_labels: Optional[List[str]] = None,
) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            colorscale=colorscale,
            zmid=zmid,
            colorbar=dict(title="weight"),
            hovertemplate=f"{x_title}=%{{x}}<br>{y_title}=%{{y}}<br>w=%{{z:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=480,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def expert_usage_overview(trace: Dict[str, Any]) -> go.Figure:
    """Layer × expert mean activation heat-map."""
    z = np.array(trace["summary"]["expert_usage"], dtype=float)
    return heatmap(
        z,
        x_title="expert",
        y_title="MoE layer",
        title="Mean expert activation (probability) per layer",
    )


def load_balance_chart(trace: Dict[str, Any]) -> go.Figure:
    load = np.array(trace["summary"]["expert_load"], dtype=float)
    ideal = trace["top_k"] / max(trace["num_experts"], 1)
    fig = go.Figure()
    fig.add_bar(x=list(range(len(load))), y=load, name="observed load")
    fig.add_hline(
        y=ideal,
        line_dash="dash",
        annotation_text=f"balanced = {ideal:.3f}",
        line_color="crimson",
    )
    fig.update_layout(
        title="Expert load distribution (selection share)",
        xaxis_title="expert id",
        yaxis_title="fraction of selections",
        height=360,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def entropy_chart(trace: Dict[str, Any]) -> go.Figure:
    ent = trace["summary"]["routing_entropy"]
    fig = go.Figure()
    fig.add_scatter(x=list(range(len(ent))), y=ent, mode="lines+markers")
    fig.update_layout(
        title="Routing entropy per layer (nats) — lower = more decisive routing",
        xaxis_title="MoE layer",
        yaxis_title="mean entropy",
        height=320,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def token_expert_heatmap(trace: Dict[str, Any], layer_index: int) -> go.Figure:
    """Per-token × per-expert routing probabilities for one layer."""
    layer = next(l for l in trace["layers"] if l["layer_index"] == layer_index)
    z = np.array(layer["router_probs"], dtype=float).T  # experts on Y, tokens on X
    return heatmap(
        z,
        x_title="token",
        y_title="expert",
        x_labels=[f"{i}:{t}" for i, t in enumerate(trace["tokens"])],
        title=f"Layer {layer_index} — per-token expert routing weights",
    )


# ──────────────────────────────────────────────────────────────────────────
# Sidebar / status
# ──────────────────────────────────────────────────────────────────────────
st.title("🧭 Llama 4 MoE — Prompt Router & Debugger")
st.caption(
    "Hook into Llama 4's sparse Mixture-of-Experts routers and see exactly "
    "which experts each token activates."
)

cfg = fetch_config()
with st.sidebar:
    st.header("Backend")
    st.code(API_URL)
    if cfg is None:
        st.error("API unreachable. Start it with `uvicorn src.api:app --port 8000`.")
    else:
        mode = cfg.get("mode", "?")
        st.success(f"Connected — mode: **{mode}**")
        st.json(cfg)
    st.divider()
    st.markdown(
        "**Tip:** set `MOE_MOCK=1` on the backend to explore the UI without "
        "downloading Llama 4 weights."
    )


# ──────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────
tab_inspect, tab_compare = st.tabs(["🔍 Inspect", "⚖️ Compare prompts"])


with tab_inspect:
    prompt = st.text_area(
        "Prompt",
        value="Explain the theory of relativity to a 10-year-old.",
        height=120,
        key="inspect_prompt",
    )
    if st.button("Run routing trace", type="primary", key="run_inspect"):
        if cfg is None:
            st.error("Backend not connected.")
        else:
            with st.spinner("Running instrumented forward pass…"):
                try:
                    trace = post_infer(prompt)
                    st.session_state["trace"] = trace
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")

    trace = st.session_state.get("trace")
    if trace:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tokens", len(trace["tokens"]))
        c2.metric("MoE layers", len(trace["layers"]))
        c3.metric("Experts (top-k)", f"{trace['num_experts']} ({trace['top_k']})")

        st.plotly_chart(expert_usage_overview(trace), use_container_width=True)

        col_a, col_b = st.columns(2)
        col_a.plotly_chart(load_balance_chart(trace), use_container_width=True)
        col_b.plotly_chart(entropy_chart(trace), use_container_width=True)

        st.subheader("Per-token drill-down")
        layer_ids = [l["layer_index"] for l in trace["layers"]]
        sel_layer = st.select_slider("MoE layer", options=layer_ids, value=layer_ids[0])
        st.plotly_chart(
            token_expert_heatmap(trace, sel_layer), use_container_width=True
        )

        with st.expander("Selected experts per token (this layer)"):
            layer = next(l for l in trace["layers"] if l["layer_index"] == sel_layer)
            rows = []
            for i, tok in enumerate(trace["tokens"]):
                if i >= len(layer["topk_indices"]):
                    break
                idxs = layer["topk_indices"][i]
                wts = layer["topk_weights"][i]
                rows.append(
                    {
                        "token": f"{i}: {tok}",
                        "experts": ", ".join(str(e) for e in idxs),
                        "weights": ", ".join(f"{w:.3f}" for w in wts),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tab_compare:
    st.markdown(
        "Run two prompts and inspect the **signed difference** in expert "
        "activation. Red = prompt B routes there *more*, blue = *less*."
    )
    cc1, cc2 = st.columns(2)
    prompt_a = cc1.text_area(
        "Prompt A", value="Write a Python function to sort a list.", height=120
    )
    prompt_b = cc2.text_area(
        "Prompt B", value="Write a haiku about the ocean.", height=120
    )

    if st.button("Compare", type="primary", key="run_compare"):
        if cfg is None:
            st.error("Backend not connected.")
        else:
            with st.spinner("Tracing both prompts…"):
                try:
                    st.session_state["compare"] = post_compare(prompt_a, prompt_b)
                except Exception as exc:
                    st.error(f"Compare failed: {exc}")

    comp = st.session_state.get("compare")
    if comp:
        diff = comp["diff"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total divergence (L1)", f"{diff['total_divergence']:.3f}")
        m2.metric("Layers compared", diff["num_layers"])
        m3.metric("Experts", diff["num_experts"])

        delta = np.array(diff["usage_delta"], dtype=float)
        st.plotly_chart(
            heatmap(
                delta,
                x_title="expert",
                y_title="MoE layer",
                colorscale="RdBu",
                zmid=0.0,
                title="Δ expert activation  (B − A)",
            ),
            use_container_width=True,
        )

        # Per-layer divergence — quickly find where routing diverges most.
        div = diff["divergence_per_layer"]
        fig = go.Figure()
        fig.add_bar(x=list(range(len(div))), y=div)
        fig.update_layout(
            title="Routing divergence per layer (Σ|Δ|)",
            xaxis_title="MoE layer",
            yaxis_title="L1 divergence",
            height=320,
            margin=dict(l=60, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        ca, cb = st.columns(2)
        ca.plotly_chart(
            heatmap(
                np.array(diff["usage_a"], dtype=float),
                x_title="expert",
                y_title="layer",
                title="Prompt A activation",
            ),
            use_container_width=True,
        )
        cb.plotly_chart(
            heatmap(
                np.array(diff["usage_b"], dtype=float),
                x_title="expert",
                y_title="layer",
                title="Prompt B activation",
            ),
            use_container_width=True,
        )
