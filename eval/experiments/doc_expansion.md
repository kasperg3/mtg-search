# EXP-005: document expansion with LLM role descriptions (cancelled)

**Status: cancelled by the user on 2026-10-14, before any encoding or evaluation.** Cards can fill several roles. A single generated "role" description per card would bias or limit an agent that uses search to design decks. This note keeps only the SparseUp k=12 findings, which apply to SparseUp's own learned expansion whether or not role text is added.

## What happened before the stop

- I wrote a generator for GLM-5.3 (Z.ai's OpenAI-compatible `https://api.z.ai/api/paas/v4/chat/completions`, model `glm-5.3`). Its input was `doc_text` only, never oracle tags. It ran a 5-card smoke test and nothing more; the 200-card pilot was never started.
- **Spend:** 5 successful calls, 1,584 prompt tokens and 1,546 completion tokens (1,302 of them reasoning tokens), about **$0.01** at $1.40 / $4.40 per 1M tokens for input / output.
- **API notes:** glm-5.3 cannot turn thinking off. `thinking: {"type": "disabled"}` returns error 1210, so the minimum is `{"type": "enabled", "effort": "low"}`. Even at that setting, reasoning was about 85% of output tokens, with 5–12 s latency per card. Extrapolating the smoke test gives about $60 for 34,646 cards, which is above the $20 budget. That figure comes from 5 cards only, so treat it as a rough indication.
- **Cleanup:** I deleted the local cache (`data/expansions/roles_glm-5.3.jsonl`) and the generator script. Nothing was uploaded to HF, no HF Jobs were launched, and no changes were made to `jobs/encode.py` or `eval/run.py`.

## SparseUp k=12: what it is

Sources: [blog](https://www.linkup.so/blog/introducing-sparseup-by-linkup), the model card, and `config.json` / `modeling_splade.py` at revision `08314498`.

- **k is per token position, not per document.** SparseUp's encoder computes MLM logits, applies `log1p(relu(x − 15))`, then keeps only the **12 largest vocab dims at each input position** (`position_top_k: 12`), then max-pools over positions and folds case and `Ġ` variants (vocab 50k → ~34k). The blog says: "this does not cap the vector's L0; it caps per-token fan-out". The k is an expansion budget per input token, chosen to make optimisation easier and to reduce stuffing. The authors call 12 "a good trade-off" but publish no ablation over k.
- **It is part of the model forward pass.** It is a config key, so it runs in `encode`, for both queries (`[Q] `, max 128) and documents (`[D] `, max 512), in training and at inference. Our existing index from `jobs/encode.py --kind sparse` (EXP-004) already has it. Nothing extra is needed to "apply k=12".
- **Measured on our index** (34,646 cards, `doc_text` p50 61 tokens), a document vector has on average 198 non-zero dims (p50 196), or about 3.4 per input token. Of those, **85% are expansion dims** (folded vocab ids absent from the card's own subtokens), carrying **76% of the weight mass**. The mean is 168 expansion terms per card (p50 166, p90 214), and no card has 12 or fewer. Expansion is therefore most of what SparseUp indexes, not a small add-on.

## Can k=12 be used as an index-time cap on expansion terms?

- **Not in the sense of "≤12 non-literal terms per document".** The model does not expose that. The cap applies per position, before pooling. A per-document cap of 12 would be a different and very aggressive pruning, removing about 156 of 168 expansion terms from a typical card. It would also push the vectors outside the distribution the model was trained on.
- **Possible ways to run it, none of them tried:**
  1. **Change `position_top_k` at encode time.** Pass a config override such as `AutoModel.from_pretrained(..., position_top_k=K)` or `None` for uncapped, then re-encode on HF Jobs (about 5.5 min on a T4). This is off-distribution for any K ≠ 12, and the query side would have to match or be deliberately left at 12.
  2. **Post-hoc pruning of the existing CSR, no re-encode.** Keep the literal dims, which are tokenizer → fold ids (the logic is in `SparseUpRetriever._literal_dims`), plus the top-N expansion dims by weight per card. This is cheap and runs locally, and it gives a direct "how much does document expansion buy us" curve for N ∈ {0, 12, 50, all}.
- If document expansion is revisited, option 2 is the cheap first experiment. It tests whether SparseUp's own learned expansion helps the vague (para/deck) sets without hurting handwritten, and it involves no LLM text and no single-role bias.
