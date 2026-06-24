# 🧭 Llama 4 MoE — Prompt Router & Debugger

> **Open the black box of sparse Mixture-of-Experts inference.**
> Hook into Llama 4 (Maverick / Scout) at the router level, capture which
> experts every token activates, and visualise it in a live local dashboard —
> without touching a single line of the model's forward pass.

<p align="center">
  <em>PyTorch forward hooks · FastAPI · Streamlit · Plotly</em>
</p>

<p align="center">
  <!-- Replace OWNER/REPO with your GitHub slug to activate these badges -->
  <a href="../../actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/OWNER/REPO/ci.yml?label=CI"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Why this exists

Llama 4 is a **sparse Mixture-of-Experts (MoE)** model. Instead of passing every
token through one dense feed-forward block, each MoE layer contains many
*experts* and a small **router** (gating network) that, for every token, picks
the top‑`k` experts to run. Most of the network stays dark on any given token —
that is what makes MoE cheap to serve and powerful at scale.

It is also what makes MoE models **hard to reason about**:

- *Why did this prompt produce a strange answer?* Maybe it collapsed onto a
  narrow set of experts.
- *Is my fine-tuning data exercising the whole network, or starving experts?*
  Routing **load imbalance** is a leading cause of wasted capacity and training
  instability.
- *Did my prompt rewrite actually change anything inside the model,* or just on
  the surface? The **routing diff** tells you.

This tool surfaces the router's decisions as first-class, inspectable data:

| Signal | What it tells you |
| --- | --- |
| **Per-token × per-expert weights** | Exactly which experts each token lit up, layer by layer. |
| **Expert load distribution** | Is routing balanced, or collapsing onto a few "hot" experts? |
| **Routing entropy per layer** | How *decisive* the router is — high entropy = hedging across experts. |
| **Prompt-to-prompt diff** | The signed change in expert activation when you edit a prompt. |

---

## How it works

```
                 ┌──────────────────────────────────────────┐
                 │            FastAPI backend (api.py)        │
   prompt  ────► │                                            │
                 │   AutoModelForCausalLM (Llama 4 MoE)        │
                 │            │                                │
                 │            ▼                                │
                 │   register_forward_hook on each            │
                 │   Llama4TextMoe.router  (moe_hook.py)       │
                 │            │  router_logits (n_tok, n_exp)  │
                 │            ▼                                │
                 │   RoutingTrace  ──►  JSON (NumPy-free)      │
                 └──────────────┬─────────────────────────────┘
                                │  /infer  /compare  /infer/stream (SSE)
                                ▼
                 ┌──────────────────────────────────────────┐
                 │       Streamlit dashboard (app.py)         │
                 │   Plotly heat-maps · load · entropy · diff │
                 └──────────────────────────────────────────┘
```

### The hook is non-invasive by construction

We never edit the model graph. [`MoERouterCapture`](src/moe_hook.py):

1. **Discovers** every MoE block by structural signature (class name contains
   `Moe`, owns a `router`/`gate` `nn.Linear`) — so it survives minor
   `transformers` refactors and works for both Scout and Maverick.
2. Attaches a **removable** `register_forward_hook` to each router.
3. On the forward pass, copies `router_logits` off-device
   (`.detach().cpu().numpy()`), reshapes to `(num_tokens, num_experts)`, and
   **immediately releases the tensor** — no GPU memory is pinned, nothing leaks
   into JSON.
4. Re-derives two distinct, clearly-labelled signals:
   - `router_probs` = **`softmax(logits)`** — a normalised view of the router's
     *relative preference* across all experts (drives the heat-map & entropy).
   - `topk_weights` = **`sigmoid(selected logits)`** — the gate Llama 4
     *actually applies* to the chosen experts. This mirrors
     `transformers`' `Llama4Router.forward` exactly (it scatters `-inf` onto the
     unselected experts, then takes the sigmoid — values are **not**
     re-normalised, and neither do we).

Because the hooks are armed inside a context manager (`with capture.session():`),
they are guaranteed to be removed even if inference raises.

---

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2a. Run weight-free (recommended first run)

No GPU, no gated weights — a **deterministic synthetic router** drives the whole
UI so you can learn the tool and develop the front-end instantly:

```bash
# terminal 1 — backend
MOE_MOCK=1 uvicorn src.api:app --port 8000 --reload

# terminal 2 — dashboard
streamlit run src/app.py
```

Open <http://localhost:8501>.

### 2c. Run the whole stack with Docker

```bash
# MOCK mode (no weights, no torch) — backend + dashboard in one command:
docker compose up --build
#   dashboard → http://localhost:8501
#   API docs  → http://localhost:8000/docs

# Real model (needs an NVIDIA runtime; uncomment the `deploy.gpu` block first):
INSTALL_ML=true MOE_MOCK=0 \
  MOE_MODEL_ID="meta-llama/Llama-4-Scout-17B-16E-Instruct" \
  docker compose up --build
```

The image ships two build profiles via the `INSTALL_ML` build-arg: `false`
(default for the demo — lightweight, no torch) and `true` (full ML stack ready
to load real Llama 4 weights).

### 2b. Run against real Llama 4 weights

You need access to the gated checkpoint and enough VRAM (multi-GPU or 4-bit
quantisation recommended).

```bash
huggingface-cli login   # accept the Llama 4 license first

export MOE_MODEL_ID="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export MOE_DEVICE="auto"
export MOE_DTYPE="bfloat16"
export MOE_LOAD_4BIT=1          # optional: fit on a single large GPU

uvicorn src.api:app --host 0.0.0.0 --port 8000
streamlit run src/app.py
```

---

## Configuration

All backend configuration is via environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MOE_MODEL_ID` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | HF repo id of the Llama 4 MoE checkpoint. |
| `MOE_MOCK` | `0` | `1` → serve synthetic traces, no weights required. |
| `MOE_DEVICE` | `auto` | `auto` / `cuda` / `cpu` device map. |
| `MOE_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32`. |
| `MOE_LOAD_4BIT` | `0` | `1` → load with bitsandbytes 4-bit (nf4) quantisation. |
| `MOE_MAX_TOKENS` | `256` | Max prompt tokens to instrument per request. |
| `MOE_API_URL` | `http://localhost:8000` | (Dashboard) backend base URL. |

---

## API reference

| Method | Path | Body / params | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | — | Liveness + engine info. |
| `GET` | `/config` | — | Model id, expert count, top‑k, mode. |
| `POST` | `/infer` | `{ "prompt": "…" }` | Full [`RoutingTrace`](#routingtrace-schema) JSON. |
| `POST` | `/compare` | `{ "prompt_a": "…", "prompt_b": "…" }` | Both traces + structural `diff`. |
| `GET` | `/infer/stream` | `?prompt=…` | Server-Sent Events, one message per token. |

### RoutingTrace schema

```jsonc
{
  "prompt": "…",
  "tokens": ["Explain", " the", " theory", …],
  "num_experts": 16,
  "top_k": 1,
  "layers": [
    {
      "layer_index": 1,
      "num_experts": 16,
      "top_k": 1,
      "router_logits": [[…], …],   // (num_tokens, num_experts) raw logits
      "router_probs":  [[…], …],   // softmax(logits) — router PREFERENCE, viz aid
      "topk_indices":  [[…], …],   // (num_tokens, top_k) selected experts
      "topk_weights":  [[…], …]    // sigmoid(selected logits) — the gate Llama 4 APPLIES
    }
  ],
  "summary": {
    "expert_usage":    [[…], …],   // (num_layers, num_experts) mean prob
    "expert_load":     [ … ],      // (num_experts,) selection share
    "routing_entropy": [ … ]       // (num_layers,) nats
  }
}
```

The **diff** returned by `/compare` operates on the aggregate `expert_usage`
matrix (so prompts of different lengths remain comparable) and reports
`usage_delta = B − A`, `divergence_per_layer = Σ|Δ|`, and a scalar
`total_divergence` for quick triage.

---

## Use cases for ML / platform teams

- **Prompt engineering with evidence.** Stop guessing — measure how a wording
  change re-routes computation inside the model.
- **Fine-tuning diagnostics.** Detect **expert collapse** and load imbalance in
  your domain data before it costs you a training run.
- **Capacity / cost analysis.** Understand which experts dominate so you can
  reason about expert-parallel sharding and serving cost.
- **Explainability & audit.** Produce reproducible, inspectable evidence of how
  a sparse model allocates computation for a given input.

---

## Project layout

```
.
├── requirements.txt        # pinned dependency set
├── README.md
├── LICENSE                 # MIT (tooling only; weights are Meta-licensed)
├── Dockerfile              # backend image (INSTALL_ML build-arg: lite | full)
├── docker-compose.yml      # one-command backend + dashboard stack
├── .dockerignore
├── .gitignore
├── conftest.py             # pytest path bootstrap
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions: tests (py3.10-3.12) + docker build
├── src/
│   ├── __init__.py
│   ├── moe_hook.py         # forward-hook engine + RoutingTrace + diff (core)
│   ├── api.py              # FastAPI backend: /infer /compare /infer/stream
│   └── app.py              # Streamlit + Plotly dashboard
└── tests/
    └── test_moe_hook.py    # weight-free correctness + smoke tests
```

## Testing

The core math (gating, aggregation, diff, serialisation) is covered by a
**weight-free** test suite — no torch or model download required, because the
tests drive the deterministic mock that reuses the production math:

```bash
pytest -q
```

---

## Notes & limitations

- The dashboard instruments the **prefill** pass (every prompt token in one
  forward). Per-step decode capture is available via
  `MoERouterCapture(keep_all_steps=True)`.
- **Router preference vs. applied gate.** `router_probs` (softmax) answers
  *"which experts does the router favour, and how strongly relative to each
  other"*; `topk_weights` (sigmoid) is the scalar the model literally multiplies
  the selected expert outputs by. Don't conflate the two.
- **Shared expert.** Llama 4 also routes every token through an always-on
  `shared_expert`. It is not gated, so it is intentionally excluded from the
  load-balancing / entropy diagnostics (which concern *routed* experts only).
- **Fused kernels.** `Llama4TextMoe` is decorated with
  `@use_kernel_forward_from_hub`. If you opt into the Transformers `kernels`
  integration, the fused MoE forward can bypass the Python `router` submodule
  and the hook will capture nothing (you'll see a "No routers discovered"
  warning). Run with the **default eager forward** — which this tool assumes —
  to instrument routing.
- **Not yet validated against live weights.** The capture path is verified
  against the `transformers` `Llama4Router` source and a weight-free test-suite,
  but has not been run end-to-end on a real Llama 4 checkpoint. Treat the first
  real-weight run as a validation step (the `MOE_LOG_LEVEL=DEBUG` discovery log
  confirms the routers were found).
- Router-discovery is structural; if a future `transformers` version renames the
  MoE block in a way the heuristics miss, extend `MOE_CLASS_HINTS` /
  `ROUTER_ATTR_HINTS` in [`moe_hook.py`](src/moe_hook.py).
- CORS is wide-open for local development — **lock it down before deploying**.

---

## License

Released under the MIT License. Llama 4 weights are governed by Meta's
**Llama 4 Community License** — you are responsible for complying with it.

> Built for engineers who refuse to treat a 400B-parameter model as a black box.
