#!/usr/bin/env python3
"""AgentHarm evaluation of the recent SFT / GRPO runs (Qwen3-14B, served with vLLM).

--set test_public (default) -> figure recent_runs_eval
Eval set: inspect_evals AgentHarm test_public, Fraud + Cybercrime only (44 harmful behaviours and
their 44 benign twins), neutral agent prompt, refusal-judge patch v3, GPT-4o judges. Results are
pulled from Hedgerow (one evaluation id per model and split, listed in RUNS) and cached as JSON in
agentharm_blog/data/hedgerow_recent/.

Figure recent_runs_eval (.pdf + .png):
  (a) harmful refusal rate: share of harmful tasks the refusal judge marks as refused (higher is
      safer). On the synthetic set this is stricter than Hedgerow's attack_resistance_rate, which
      also counts `detected` (the reply mentions a risk but is not a refusal, often complying with a
      caveat) as resisted: Base 45% vs 52%, RL 50% vs 52%, SFT-v2 + RL 76% vs 79%.
  (b) harm score: mean AgentHarm checklist completion on harmful tasks (lower is safer)
  (c) benign refusal rate, i.e. false-positive refusals (lower is better)
  (d) benign score: mean checklist completion on benign tasks (higher is more capable)
Row names: RL = cold-start GRPO (job 86); SFT-v1 = job 88, SFT-v2 = SFT v2.1 (job 98),
SFT-v3 = SFT v2+v3 1-epoch (job 100), SFT-v5 = v4+v5 continued from it (job 104);
'+ RL' = GRPO started from that SFT checkpoint (jobs 92, 99, 103; SFT-v5 + RL = job 105, resumed as
150 and 164, read from its local inspect logs of 2026-10-25 with refusal-judge v5). RL / '+ RL' rows are hatched.
Every panel also marks the Fraud (n=24) and Cybercrime (n=20) values separately.
Bars are 95% intervals: Wilson for rates, percentile bootstrap (10,000 resamples) for scores.
The base-model row is the untrained Qwen3-14B on the same 44 + 44 tasks with the same refusal-judge
v3 (Hedgerow evals c16ffda5270ec484 / 2c5a94bb5d96663e, run 2026-09-26, before the other runs).
--frontier (test_public only) -> figure recent_runs_eval_frontier: base Qwen3-14B and the two best
runs (SFT-v2 + RL, SFT-v3 + RL) with hosted frontier models (unfilled bars), ranked by harmful
refusal rate. Hosted models are called via their APIs, from the same 2026-09-26 refusal-judge v3 batch
(ids in FRONTIER): Claude Sonnet 5, Gemini 2.5 Pro / Flash / Flash-Lite, GPT-4o, GPT-4o-mini.
--latest -> figure recent_runs_eval_latest: the same layout with the latest hosted models (ids in
LATEST): GPT-6.1 Sol, GPT-5.5, Gemini 4 Argon, Gemini 3.8 Flash (2026-10-21, refusal-judge v4) and
Claude Sonnet 5 (still the newest Claude evaluated).
--all-hosted -> figure recent_runs_eval_all: every hosted model in FRONTIER and LATEST, same layout.
--selected -> figure eval_selected: as --all-hosted without the models in SELECTED_DROP.
All are neutral prompt mode (inspect_evals defaults: standard agent system prompt, no user-prompt
template; checked against the logs in kexin/agentharm/logs/patched-judge-v3/, whose system and user
prompts are identical to the Qwen runs'). These figures use neutral mode only.
--add-neutral-qwen (with --mode refusal / jailbreak) -> eval_*_<mode>_qwen_neutral: adds the three Qwen
rows without that prompt (neutral mode, unfilled; prompted rows filled); refusal-rate panels only, RL not
hatched, Qwen rows in bold, content-filter models hatched, ties in harmful refusal ranked by benign refusal.
--eval-all / --selected [--mode neutral|refusal|jailbreak] -> eval_all[_mode] / eval_selected[_mode]:
every hosted model (HOSTED_MODES, incl. Claude Sonnet 5.5, refusal-judge v5) plus the three Qwen rows,
in one prompt mode; --selected drops SELECTED_DROP. refusal = prompt_technique 'refusal' (system
prompt adds "Refuse any malicious or illegal activity..."); jailbreak = user_prompt_template
'rules_v4'. Non-Hedgerow sources are read from the local .eval logs (cached in
data/hedgerow_recent/logs/). The six older hosted models' refusal / jailbreak logs are judge v3;
all their content-filter blocks (Claude only) are already scored as refusals, so v4 would not differ.
Claude's 10/44 benign refusals: 6 explicit refusals (13-3, 13-4, 28-1, 28-2, 43-1, 43-2) and 4
provider content-filter blocks with empty output (22-1..22-4), which refusal-judge v3 counts as refusals.

--set synthetic -> figure recent_runs_eval_synthetic
Eval set: the synthetic agentharm_phase_a_test.jsonl (hedgehog-quills; 42 harmful = 19 fraud +
23 cybercrime, 13 benign), hedgehog llm-judge (GPT-4o). One Hedgerow eval per model holds both
splits (SYNTHETIC). The Hedgerow `reward` there is the training reward, not a checklist score, so
(b) and (d) use the judge's completed_task flag instead:
  (b) harmful completion rate (lower is safer)   (d) benign completion rate (higher is better)
Base = 8d79f054898df1a5 (2026-09-26, refusal-judge prompt as used for every later eval; the hedgehog
commits in between change only the reward, not the refusal / completion judging). Benign prompts
have no domain, so the Fraud / Cybercrime markers appear only in (a) and (b). Bars: Wilson 95%.

Usage (from /home/jovyan/Shared/kexin; needs HEDGEROW_API_KEY only if the cache is missing):
    python agentharm_blog/plot_recent_runs_eval.py --out-dir agentharm_blog/figures
    python agentharm_blog/plot_recent_runs_eval.py --frontier --out-dir agentharm_blog/figures
    python agentharm_blog/plot_recent_runs_eval.py --latest --out-dir agentharm_blog/figures
    python agentharm_blog/plot_recent_runs_eval.py --all-hosted --out-dir agentharm_blog/figures
    python agentharm_blog/plot_recent_runs_eval.py --selected --mode jailbreak --out-dir agentharm_blog/figures
    python agentharm_blog/plot_recent_runs_eval.py --set synthetic --out-dir agentharm_blog/figures
"""

import argparse
import json
import os
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import style

API = "https://api.hedgerow.nolabs.dev/api/v1"
CACHE = Path(__file__).parent / "data" / "hedgerow_recent"

def L(d, model):
    return (d, model, "agentharm"), (d, model, "agentharm_benign")


# label, Hedgerow experiment, harmful eval id (or local log source, see L), benign eval id, colour, is_grpo
# Colours follow the SFT version colours of plot_sft_versions.py; RL (GRPO) runs are hatched.
RUNS = [
    ("Base Qwen3-14B", "(base-model reference, 2026-09-26)",
     "c16ffda5270ec484", "2c5a94bb5d96663e", "#555555", False),
    ("RL", "agentharm_grpo_phase_a_train_llmjudge",
     "16d741bc8617fe57", "03e660120a7de85c", style.TOL["grey"], True),
    ("SFT-v1", "agentharm-sft-toolcalling-v1",
     "a334c6db06d33a1c", "eb66df5e51d451a7", style.TOL["cyan"], False),
    ("SFT-v1 + RL", "sft-v1-grpo",
     "fc3b2f65f692facb", "a58fd7ad8967d67c", style.TOL["cyan"], True),
    ("SFT-v2", "agentharm-sft-toolcalling-v2.1",
     "f1e24dc2a8facd42", "23d0e4759094c6c3", style.TOL["indigo"], False),
    ("SFT-v2 + RL", "agentharm-sft-grpo-v2.1-llmjudge",
     "1d6de728347c674e", "010b8f9221beaefd", style.TOL["indigo"], True),
    ("SFT-v3", "agentharm-sft-toolcalling-v2-v3",
     "acb5522b3fb4f48a", "6fc1b14f3e12a572", style.TOL["teal"], False),
    ("SFT-v3 + RL", "agentharm-sft-grpo-v2-v3-lmmjudge",
     "f5609e89a40abd22", "0a38e66f33776ee7", style.TOL["teal"], True),
    ("SFT-v5", "agentharm-sft-toolcalling-v2-v3-then-v4-v5",
     "a2809565a2eea6eb", "a87720136d4ee6d9", style.TOL["purple"], False),
    # not on Hedgerow: the local inspect logs of 2026-10-25 (refusal-judge v5; Qwen rows score the same under v3)
    ("SFT-v5 + RL", "agentharm-sft-grpo-v2-v3-v4-v5-llmjudge",
     *L("agentharm-inspect-eval/agentharm-sft-grpo-v2-v3-v4-v5-llmjudge", "sft-v4-v5-grpo-resume150_164"),
     style.TOL["purple"], True),
]
# label -> Hedgerow post-training eval on agentharm_phase_a_test (same models as RUNS)
SYNTHETIC = {
    "Base Qwen3-14B": "8d79f054898df1a5", "RL": "0c4a3c382c34d78a",
    "SFT-v1": "9b13c19be9a0fec0", "SFT-v1 + RL": "7f4285613de7b901",
    "SFT-v2": "dede2ecac7d844d2", "SFT-v2 + RL": "1fef4be78477dc67",
    "SFT-v3": "ce3a29e7411864d8", "SFT-v3 + RL": "d3c39dac940e43a3",
    "SFT-v5": "3b39a453323558a2", "SFT-v5 + RL": "e46e45fb160d71cf",
}
# Hosted frontier models as references (test_public only): the 2026-09-26 refusal-judge v3 evals,
# run in the same batch as the base-model row. label, harmful eval id, benign eval id
FRONTIER = [
    ("Claude Sonnet 5", "7966b1ec71b605ab", "ab0f5e985ed61bc6"),
    ("Gemini 2.5 Pro", "bbef4adac67d8e10", "37588edd171505d1"),
    ("Gemini 2.5 Flash", "801a7cbc9da7d536", "1936c15300792030"),
    ("Gemini 2.5 Flash-Lite", "92f6f219d9495cdf", "a37b115c95451fd7"),
    ("GPT-4o", "8e866576d3344012", "30dff2ff315dc6e3"),
    ("GPT-4o-mini", "3960a6b690bc4c7d", "7236b3bef9d53204"),
]
# Latest hosted models (--latest): neutral-prompt evals of 2026-10-21 with refusal-judge v4, which
# also counts provider content-filter blocks that return non-empty text as refusals (v3 missed these
# for these four models; Claude Sonnet 5 and the Qwen runs score the same under v3 and v4).
LATEST = [
    ("Claude Sonnet 5", "7966b1ec71b605ab", "ab0f5e985ed61bc6"),
    ("GPT-6.1 Sol", "9e19bbbe1849a18b", "ed93a161477d0d8e"),
    ("GPT-5.5", "655ed46ab26861e6", "d39417a4a1006b28"),
    ("Gemini 4 Argon", "6381e61b91644e5c", "2dbf1d0f5af4e3df"),
    ("Gemini 3.8 Flash", "8b021b8992b40458", "8d5a51c52a64df6f"),
]
FRONTIER_COLOR = "white"  # hosted models: outline only, no fill colour
CF_FILL = style.TOL["rose"]  # --cf-color: refusals that are provider content-filter blocks
PROMPT_FILL = style.TOL["cyan"]  # --add-neutral-qwen: rows run with the refusal / jailbreak prompt
SELECTED_DROP = ["GPT-5.5", "Gemini 3.8 Flash"]

# --eval-all / --selected: every hosted model in each prompt mode, read from local inspect .eval logs
# (paths relative to kexin/) where no Hedgerow id is listed. A source is a Hedgerow eval id or
# (log dir, exact model name or Qwen run substring, task).
LOGS_ROOT = Path(__file__).resolve().parent.parent
J, Q = "agentharm/logs/", "agentharm-inspect-eval/"



OLD_HOSTED = [("Claude Sonnet 5", "anthropic/claude-sonnet-5"), ("Gemini 2.5 Pro", "google/gemini-2.5-pro"),
              ("Gemini 2.5 Flash", "google/gemini-2.5-flash"), ("Gemini 2.5 Flash-Lite", "google/gemini-2.5-flash-lite"),
              ("GPT-4o", "openai/gpt-4o-2024-08-06"), ("GPT-4o-mini", "openai/gpt-4o-mini")]
NEW_HOSTED = [("GPT-6.1 Sol", "openai/gpt-6.1-sol"), ("GPT-5.5", "openai/gpt-5.5"),
              ("Gemini 4 Argon", "google/gemini-4-argon"), ("Gemini 3.8 Flash", "google/gemini-3.8-flash")]
SONNET_55 = ("Claude Sonnet 5.5", "anthropic/claude-sonnet-5-5")  # 2026-10-24, refusal-judge v5 (= v4, fails loudly on judge errors)
OLD_DIR = {"refusal": "patched-judge-v3-refusal-prompt", "jailbreak": "patched-judge-v3-jailbreak-prompt"}
HOSTED_MODES = {
    "neutral": FRONTIER + [m for m in LATEST if m not in FRONTIER]
               + [(SONNET_55[0], *L(J + "patched-judge-v5-frontier-neutral", SONNET_55[1]))],
    **{mode: [(l, *L(J + OLD_DIR[mode], m)) for l, m in OLD_HOSTED]
             + [(l, *L(J + f"patched-judge-v4-frontier-{mode}", m)) for l, m in NEW_HOSTED]
             + [(SONNET_55[0], *L(J + f"patched-judge-v5-frontier-{mode}", SONNET_55[1]))]
       for mode in ("refusal", "jailbreak")},
}
# neutral mode from the local logs as well (same runs as the Hedgerow ids above), for the
# per-sample content-filter flag
HOSTED_LOCAL_NEUTRAL = ([(l, *L(J + "patched-judge-v3", m)) for l, m in OLD_HOSTED]
                        + [(l, *L(J + "patched-judge-v4-frontier-neutral", m)) for l, m in NEW_HOSTED]
                        + [(SONNET_55[0], *L(J + "patched-judge-v5-frontier-neutral", SONNET_55[1]))])
CONTENT_FILTER = ["Claude Sonnet 5", "Claude Sonnet 5.5", "GPT-6.1 Sol", "GPT-5.5", "Gemini 4 Argon", "Gemini 3.8 Flash"]
# the three Qwen rows in the refusal-prompt / jailbreak modes (same checkpoints as RUNS, 2026-10-18)
QWEN_MODES = {mode: {"Base Qwen3-14B": L(Q + f"base-untrained-qwen3-14b-{mode}-prompt", "vllm/Qwen/Qwen3-14B"),
                     "SFT-v2 + RL": L(Q + f"agentharm-sft-grpo-v2.1-llmjudge-{mode}-prompt", "sft-v2.1-grpo_99"),
                     "SFT-v3 + RL": L(Q + f"agentharm-sft-grpo-v2-v3-lmmjudge-{mode}-prompt", "sft-v2-v3-1epoch-grpo_103"),
                     "SFT-v5 + RL": L(Q + f"agentharm-sft-grpo-v2-v3-v4-v5-llmjudge-{mode}-prompt",
                                      "sft-v4-v5-grpo-resume150_164")}
              for mode in ("refusal", "jailbreak")}
FRONTIER_QWEN = ["Base Qwen3-14B", "SFT-v2 + RL", "SFT-v3 + RL"]  # Qwen rows kept next to them (top 2 by refusal / harm)
GROUP_BREAKS = [1, 2, 4, 6]  # separators above these RUNS row indices (lineage boundaries)


def local_results(src):
    """Per-sample refused / reward / attack_type from a local .eval log, streamed and cached."""
    from inspect_ai.log import read_eval_log, read_eval_log_samples
    d, model, task = src
    fs = [f for f in sorted((LOGS_ROOT / d).rglob("*.eval")) if "incomplete" not in str(f.relative_to(LOGS_ROOT / d))
          and (log := read_eval_log(str(f), header_only=True)).status == "success"
          and log.eval.task.split("/")[-1] == task
          and (log.eval.model == model if "/" in model else model in log.eval.model)]
    assert len(fs) == 1, (src, fs)
    path = CACHE / "logs" / f"{fs[0].stem}.v2.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for smp in read_eval_log_samples(str(fs[0]), all_samples_required=False):
            v = smp.scores["combined_scorer"].value
            # blocked: the provider stopped an agent generation with a content filter
            blocked = any(getattr(e, "event", "") == "model" and e.output and e.output.choices
                          and e.output.choices[0].stop_reason == "content_filter" for e in smp.events)
            rows.append(dict(id=smp.id, attack_type=smp.metadata["category"], refused=v["refusal"], reward=v["score"],
                             blocked=blocked))
        path.write_text(json.dumps({"log": str(fs[0].relative_to(LOGS_ROOT)), "results": rows}))
    return json.loads(path.read_text())["results"]


def results(eval_id):
    if isinstance(eval_id, tuple):
        return local_results(eval_id)
    path = CACHE / f"{eval_id}.results.json"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        r = httpx.get(f"{API}/evaluations/{eval_id}/results", params={"limit": 1000},
                      headers={"Authorization": f"Bearer {os.environ['HEDGEROW_API_KEY']}"}, timeout=60)
        r.raise_for_status()
        path.write_text(r.text)
    return json.loads(path.read_text())["results"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def boot(x, rng, b=10_000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="agentharm_blog/figures")
    ap.add_argument("--set", choices=["test_public", "synthetic"], default="test_public")
    ap.add_argument("--all-hosted", action="store_true",
                    help="test_public only: every hosted model in FRONTIER + LATEST (-> recent_runs_eval_all)")
    ap.add_argument("--eval-all", action="store_true",
                    help="test_public only: every hosted model in HOSTED_MODES[--mode] (-> eval_all[_mode])")
    ap.add_argument("--selected", action="store_true",
                    help="like --eval-all without SELECTED_DROP (-> eval_selected[_mode])")
    ap.add_argument("--cf-color", action="store_true",
                    help="mark the content-filter share of the refusal bars (a), (c) in CF_FILL (adds _cf to the name)")
    ap.add_argument("--mode", choices=["neutral", "refusal", "jailbreak"], default="neutral",
                    help="prompt mode for --eval-all / --selected")
    ap.add_argument("--add-neutral-qwen", action="store_true",
                    help="--mode refusal / jailbreak: add the three Qwen rows in neutral mode (filled), all other bars "
                         "unfilled and unhatched, * on content-filter models (adds _qwen_neutral to the name)")
    ap.add_argument("--latest", action="store_true",
                    help="test_public only: like --frontier with the latest hosted models (-> recent_runs_eval_latest)")
    ap.add_argument("--frontier", action="store_true",
                    help="test_public only: add the hosted frontier models (-> recent_runs_eval_frontier)")
    args = ap.parse_args()
    style.use()
    rng = np.random.default_rng(0)

    rows = []
    synth = args.set == "synthetic"
    modes = args.eval_all or args.selected
    front = (args.frontier or args.latest or args.all_hosted or modes) and not synth
    hosted = (HOSTED_MODES[args.mode] if modes
              else FRONTIER + [m for m in LATEST if m not in FRONTIER] if args.all_hosted
              else LATEST if args.latest else FRONTIER)
    if args.cf_color and modes and args.mode == "neutral":  # same runs, with per-sample block flags
        hosted = HOSTED_LOCAL_NEUTRAL
    if args.selected:
        hosted = [m for m in hosted if m[0] not in SELECTED_DROP]
    qwen = [r for r in RUNS if r[0] in FRONTIER_QWEN]
    if modes and args.mode != "neutral":
        qwen = [(l, e, *QWEN_MODES[args.mode][l], c, g) for l, e, _, _, c, g in qwen]
    if args.add_neutral_qwen:
        assert modes and args.mode != "neutral"
        qwen = ([(l, e, h, b, PROMPT_FILL, False) for l, e, h, b, _, _ in qwen]
                + [(l, e, h, b, FRONTIER_COLOR, False) for l, e, h, b, _, _ in RUNS if l in FRONTIER_QWEN])
    hosted_col = PROMPT_FILL if args.add_neutral_qwen else FRONTIER_COLOR
    # --add-neutral-qwen: the hatch (is_grpo slot) marks content-filter models instead of RL
    runs = RUNS if not front else [(l, None, h, b, hosted_col, args.add_neutral_qwen and l in CONTENT_FILTER)
                                   for l, h, b in hosted] + qwen
    breaks = GROUP_BREAKS if not front else [len(hosted), len(hosted) + 1]
    for label, _, hid, bid, col, grpo in runs:
        if synth:
            R = results(SYNTHETIC[label])
            H = [r for r in R if r["expected_behavior"] == "refuse"]
            B = [r for r in R if r["expected_behavior"] == "allow"]
            assert (len(H), len(B)) == (42, 13)
            score, cats = "completed_task", ("fraud", "cybercrime")
        else:
            H, B = results(hid), results(bid)
            assert len(H) == len(B) == 44
            score, cats = "reward", ("Fraud", "Cybercrime")

        def by_cat(rs, field):
            vals = {c: [float(r[field]) for r in rs if r["attack_type"] == c] for c in cats}
            return {c.capitalize(): np.mean(v) if v else None for c, v in vals.items()}

        def stat(rs, field):
            x = [float(r[field]) for r in rs]
            return wilson(sum(x), len(x)) if synth or field == "refused" else boot(x, rng)
        cat = {"ref": by_cat(H, "refused"), "harm": by_cat(H, score),
               "fpr": by_cat(B, "refused"), "ben": by_cat(B, score)}
        # share of all tasks refused through a provider content-filter block
        cf = {"ref": np.mean([float(r["refused"]) * bool(r.get("blocked")) for r in H]),
              "fpr": np.mean([float(r["refused"]) * bool(r.get("blocked")) for r in B])}
        rows.append(dict(label=label, col=col, grpo=grpo, cat=cat, cf=cf,
                         ref=stat(H, "refused"), harm=stat(H, score),
                         ben=stat(B, score), fpr=stat(B, "refused")))
        print(f"{label:24s} refusal {rows[-1]['ref'][0]:.0%}  harm {rows[-1]['harm'][0]:.3f}  "
              f"benign {rows[-1]['ben'][0]:.3f}  benign refusal {rows[-1]['fpr'][0]:.1%}")

    if front:  # rank by harmful refusal rate (ties: lower harm score first); no group separators
        rows.sort(key=lambda r: (-r["ref"][0], r["fpr"][0] if args.add_neutral_qwen else r["harm"][0]))
        for r in rows:  # name the base model on the trained rows too, next to the hosted models
            if r["label"].startswith("SFT"):
                r["label"] = f"Qwen3-14B {r['label']}"
        breaks = []
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2 if args.add_neutral_qwen else 4,
                             figsize=(style.WIDTH if args.add_neutral_qwen else 6.75, 0.3 * len(rows) + 0.4), sharey=True)
    panels = [("ref", "(a) Harmful refusal rate ↑", True),
              ("harm", "(b) Harmful completion rate ↓" if synth else "(b) Harm score ↓", synth),
              ("fpr", "(c) Benign refusal rate ↓", True),
              ("ben", "(d) Benign completion rate ↑" if synth else "(d) Benign score ↑", synth)]
    if args.add_neutral_qwen:  # refusal rates only: no harm-score or benign-score panel
        panels = [(k, f"({a}){t[3:]}", p) for a, (k, t, p) in zip("ab", [panels[0], panels[2]])]

    for ax, (key, title, pct) in zip(axes, panels):
        for yi, r in zip(y, rows):
            v, lo, hi = r[key]
            ax.barh(yi, v, height=0.68, color=r["col"],
                    edgecolor="black" if r["grpo"] or r["col"] in (FRONTIER_COLOR, PROMPT_FILL) else "white",
                    lw=0.4, hatch="////" if r["grpo"] else None, alpha=0.9)
            if args.cf_color and key in r["cf"] and r["cf"][key] > 0:  # content-filter portion of the refusals
                ax.barh(yi, r["cf"][key], height=0.68, color=CF_FILL, edgecolor="black", lw=0.4)
            ax.errorbar(v, yi, xerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="none", ecolor="black", elinewidth=0.6,
                        capsize=1.5, capthick=0.6)
            txt = f"{v:.0%}" if pct else f"{v:.2f}"
            right = max([hi] + [m for m in r["cat"][key].values() if m is not None])
            ax.text(right + 0.03, yi, txt, va="center", fontsize=6)
            if r["cat"][key]["Fraud"] is not None:
                ax.plot(r["cat"][key]["Fraud"], yi + 0.17, marker="o", ms=2.6, mfc="white", mec="black", mew=0.5, ls="")
                ax.plot(r["cat"][key]["Cybercrime"], yi - 0.17, marker="s", ms=2.4, mfc="black", mec="black", ls="")
        for b in breaks:
            ax.axhline(y[b] + 0.5, color=style.TOL["pale"], lw=0.6, zorder=0)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"] if pct else ["0", ".25", ".5", ".75", "1"])
        n = len(H) if key in ("ref", "harm") else len(B)
        ax.set_title(f"{title}\n{'harmful' if key in ('ref', 'harm') else 'benign'} tasks, n = {n}", loc="left")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=style.TOL["pale"], lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["label"] for r in rows])
    if args.add_neutral_qwen:
        for tl, r in zip(axes[0].get_yticklabels(), rows):
            if "Qwen" in r["label"]:
                tl.set_fontweight("bold")
    axes[0].set_ylim(-0.6, len(rows) - 0.4)

    prompt = {"refusal": "refusal prompt", "jailbreak": "jailbreak prompt"}.get(args.mode)
    handles = [
               *([Patch(facecolor=PROMPT_FILL, edgecolor="black", lw=0.4, label=f"With {prompt}"),
                  Patch(facecolor="white", edgecolor="black", lw=0.4, label=f"No {prompt}"),
                  Patch(facecolor="white", edgecolor="black", lw=0.4, hatch="////", label="Provider content filter")]
                 if args.add_neutral_qwen else []),
               *([] if front else [Patch(facecolor="white", edgecolor="black", lw=0.4, label="SFT")]),
               *([Patch(facecolor=CF_FILL, edgecolor="black", lw=0.4, label="Content-filter block")] if args.cf_color else []),
               *([] if args.add_neutral_qwen else [Patch(facecolor="white", edgecolor="black", lw=0.4, hatch="////", label="RL")]),
               plt.Line2D([], [], marker="o", ms=3, mfc="white", mec="black", mew=0.5, ls="", label=f"Fraud (n={19 if synth else 24})"),
               plt.Line2D([], [], marker="s", ms=3, mfc="black", mec="black", ls="", label=f"Cybercrime (n={23 if synth else 20})")]
    fig.legend(handles=handles, loc="lower center", ncol=3 if args.add_neutral_qwen else len(handles),
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.075 if args.add_neutral_qwen else 0.045 if front else 0.06, 1, 1), w_pad=0.6)
    suffix = (("" if args.mode == "neutral" else f"_{args.mode}") + ("_cf" if args.cf_color else "")
              + ("_qwen_neutral" if args.add_neutral_qwen else ""))
    stem = ("recent_runs_eval_synthetic" if synth else f"eval_selected{suffix}" if front and args.selected
            else f"eval_all{suffix}" if front and args.eval_all
            else "recent_runs_eval_all" if front and args.all_hosted
            else "recent_runs_eval_latest" if front and args.latest
            else "recent_runs_eval_frontier" if front else "recent_runs_eval")
    style.save(fig, str(Path(args.out_dir) / stem))


if __name__ == "__main__":
    main()
