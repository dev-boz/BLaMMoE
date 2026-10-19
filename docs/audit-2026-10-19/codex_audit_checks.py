"""Independent CPU-only probes for AUDIT.md; does not change library code."""
import bisect
import contextlib
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

import torch
import yaml

from blammoe.stats import bootstrap_ci
from blammoe.toy import setup
from blammoe.toy.audit import prefix_inputs
from blammoe.toy.batching import make_batch
from blammoe.toy.bridge import transplant_hook
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.corpus import MixedSampler, Sample, recall_sample, split_recall_sample
from blammoe.toy.fused import FusedPair, Link, expert_attention_mass, run_pair, token_ends
from blammoe.toy.fused_train import general_eval_set, mixed_draw, split_eval_set
from blammoe.toy.moe import ToyMoEConfig, ToyMoELM, save
from blammoe.toy.tokenizers import BPETokenizer, CharTokenizer

torch.set_num_threads(2)
torch.manual_seed(0)
OUT = Path("audit_artifacts")
OUT.mkdir(exist_ok=True)
REPORT = {}


def record(name, value):
    REPORT[name] = value
    print(name, json.dumps(value), flush=True)


def tiny(vocab, d=16, seed=0, layers=2, experts=4, length=256):
    torch.manual_seed(seed)
    return ToyMoELM(ToyMoEConfig(vocab_size=vocab, d_model=d, n_layers=layers,
                               n_heads=2, n_experts=experts, top_k=2,
                               d_expert=16, max_len=length)).eval()


# A proper swap cannot affect a token which never routes to the replaced expert.
B = tiny(97)
moe = B.layers[0].moe
x = torch.zeros(1, 1, 16)
x[..., 0] = 1
with torch.no_grad():
    moe.router.weight.zero_()
    moe.router.weight[:, 0] = torch.tensor([3., 2., 1., 0.])
    before = moe(x)
    with transplant_hook(B, 0, 3, torch.ones_like(x), 1.):
        actual = moe(x)
    p = moe.router(x).softmax(-1)
record("transplant_unrouted", {
    "selected_experts": p.topk(2, -1).indices.tolist(),
    "replaced_expert": 3,
    "proper_replacement_max_delta": 0.0,
    "implemented_max_delta": float((actual - before).abs().max()),
})

# All expert vectors are zero, but expert identity alone transmits ten labels.
lk = Link(10, 10, 10, 1, 10, 10)
with torch.no_grad():
    for p in lk.parameters():
        p.zero_()
    lk.eid_b.weight.copy_(torch.eye(10))
    lk.rx_a.gate.fill_(1.)
    lk.rx_a.attn.in_proj_weight[20:].copy_(torch.eye(10))
    lk.rx_a.attn.out_proj.weight.copy_(torch.eye(10))
    lk.rx_a.out.weight.copy_(torch.eye(10))
    labels = torch.arange(10)
    zeros = torch.zeros(10, 1, 10)
    slots = torch.zeros(10, 1, 1, 10)
    idx = labels[:, None, None]
    mask = torch.zeros(10, 1, 2, dtype=torch.bool)
    ya, _ = lk(zeros, zeros, slots, slots, mask, mask, idx, idx)
record("identity_only_bus", {"expert_vectors_all_zero": True,
                             "decoded_labels": ya[:, 0].argmax(-1).tolist(),
                             "correct_of_10": int((ya[:, 0].argmax(-1) == labels).sum())})

# Uniform attention over all causally available entries is labelled '2x' selectivity.
stats = expert_attention_mass(torch.tensor([[[.5, .5, 0.]]]),
                              torch.tensor([[[0], [1]]]), 2,
                              torch.tensor([[True]]))
record("uniform_causal_attention", stats)
record("bootstrap_empty_resamples", bootstrap_ci([[0, 0], [1, 1]], n_boot=10000))

TEXT = "def resolve(path, results):\n    return the_results + thx + path.resolve()\n"
ta = CharTokenizer()
tb = BPETokenizer.train([recall_sample(random.Random(i), 256).text for i in range(40)]
                        + [TEXT * 3] * 20, 220)

# Token stability under adversarially extended prefixes, independently of closed masks.
rng = random.Random(381)
texts = [TEXT, "alpha     beta12\n\n  omega!", "abc<pad>def 123"]
texts += ["".join(rng.choice("abcXYZ  \n,.!?0123456789") for _ in range(80)) for _ in range(50)]
stable_checks = 0
for text in texts:
    ids, offsets = tb.encode(text)
    closed = tb.closed_ends(text, offsets)
    for cut in range(1, len(text)):
        pre_ids, pre_off = tb.encode(text[:cut])
        for j, c in enumerate(closed):
            if c <= cut:
                assert j < len(pre_ids) and pre_ids[j] == ids[j] and pre_off[j] == offsets[j]
                stable_checks += 1
record("closed_token_stability", {"texts": len(texts), "stable_token_prefix_checks": stable_checks})


def prefix_logits(pair, text_a, text_b, positions):
    rows = []
    for start in range(0, len(positions), 32):
        pos = positions[start:start + 32]
        ia, ea, _, _, _, last, ca, _ = prefix_inputs(text_a, pos, ta, tb)
        _, _, ib, eb, _, _, _, cb = prefix_inputs(text_b, pos, ta, tb)
        kw = {"write_ends_a": ca, "write_ends_b": cb} if pair.safe_mask else {}
        with torch.no_grad():
            logits, _ = pair(ia, ib, ea, eb, **kw)
        rows.append(logits[torch.arange(len(last)), last])
    return torch.cat(rows)


# Test all A logits rather than only gold NLL or pooled means; live links at every depth.
sp = split_recall_sample(random.Random(2025), 256)
cases = [("general", TEXT, TEXT), ("split", sp.view_a, sp.view_b)]
results = {}
for signal in ("experts", "resid", "moe"):
    pair = FusedPair(tiny(ta.vocab_size), tiny(tb.vocab_size, 12, 1), [0, 1],
                     d_bus=8, n_heads=2, signal=signal, rel_pos=True, safe_mask=True)
    with torch.no_grad():
        for link in pair.links.values():
            link.rx_a.gate.fill_(1.)
            link.rx_b.gate.fill_(1.)
            link.rx_a.rel.weight.normal_(0, .2)
            link.rx_b.rel.weight.normal_(0, .2)
    for name, text_a, text_b in cases:
        a = make_batch([Sample(text_a, text_a)], ta, 256)
        b = make_batch([Sample(text_b, text_b)], tb, 256)
        pos = list(range(len(text_a) - 1))
        kwargs_list = [("normal", {})]
        if signal == "experts":
            kwargs_list += [("hide", {"hide_experts": (frozenset({0}), frozenset({2}))})]
        for variant, kwargs in kwargs_list:
            with torch.no_grad():
                full_a, full_b = run_pair(pair, a, b, **kwargs)
            if not kwargs:
                honest = prefix_logits(pair, text_a, text_b, pos)
            else:
                # Explicit, separate prefix loops also exercise the hide masks.
                rows = []
                with torch.no_grad():
                    for p in pos:
                        ia, ea, _, _, _, last, ca, _ = prefix_inputs(text_a, [p], ta, tb)
                        _, _, ib, eb, _, _, _, cb = prefix_inputs(text_b, [p], ta, tb)
                        la, _ = pair(ia, ib, ea, eb, write_ends_a=ca, write_ends_b=cb, **kwargs)
                        rows.append(la[0, last[0]])
                honest = torch.stack(rows)
            delta = float((full_a[0, pos] - honest).abs().max())
            assert delta < 1e-5, (signal, name, variant, delta)
            # B's own prefix is fixed in token coordinates; remove all later A chars.
            b_deltas = []
            with torch.no_grad():
                for j, (_, end) in enumerate(b.offsets[0]):
                    ia = a.ids[:, :end]
                    ib = b.ids[:, :j+1]
                    ea = token_ends(a)[:, :end]
                    eb = token_ends(b)[:, :j+1]
                    # Closed ends derived from the independently truncated raw views.
                    a_off = a.offsets[0][:end]
                    b_off = b.offsets[0][:j+1]
                    ca = torch.tensor([ta.closed_ends(text_a[:end], a_off)])
                    cb = torch.tensor([tb.closed_ends(text_b[:end], b_off)])
                    _, lb = pair(ia, ib, ea, eb, write_ends_a=ca, write_ends_b=cb, **kwargs)
                    b_deltas.append(float((full_b[0, j] - lb[0, -1]).abs().max()))
            assert max(b_deltas) < 1e-5
            results[f"{signal}/{name}/{variant}"] = {
                "A_positions": len(pos), "A_max_logit_delta": delta,
                "B_positions": len(b_deltas), "B_max_logit_delta": max(b_deltas)}
record("safe_causality", results)

# Demonstrate that safe_mask=True on the object does not protect direct forward calls.
pair = FusedPair(tiny(ta.vocab_size), tiny(tb.vocab_size, 12, 1), [0, 1],
                 d_bus=8, n_heads=2, signal="experts", rel_pos=True, safe_mask=True)
with torch.no_grad():
    for link in pair.links.values():
        link.rx_a.gate.fill_(1.)
        link.rx_b.gate.fill_(1.)
    s = Sample(TEXT, TEXT)
    a, b = make_batch([s], ta, 256), make_batch([s], tb, 256)
    safe, _ = run_pair(pair, a, b)
    unsafe, _ = pair(a.ids, b.ids, token_ends(a), token_ends(b))
record("safe_flag_direct_forward", {"safe_mask": pair.safe_mask,
                                    "max_logit_delta": float((safe - unsafe).abs().max())})

# Count exact evaluation windows covered by the actual fusion training text stream.
cfg = ToyPOCConfig.from_yaml("configs/toy_poc.yaml")
base = setup.sampler(cfg, seed=0)
gen = general_eval_set(base.prose, base.code, 256)
draw = mixed_draw(base.prose, base.code, 256, .5, 4242)
seen = {"prose": [], "code": []}
recall_train = set()
original_window = MixedSampler._window


def tracked_window(self, src, kind):
    index = self.rng.randrange(0, len(src) - self.length)
    text = src[index:index + self.length]
    seen[kind].append((index, index + self.length))
    return Sample(text, text, kind=kind)


MixedSampler._window = tracked_window
for _ in range(3000):
    for a, b in draw(16):
        if a.kind == "split_recall":
            recall_train.add(a.text)
MixedSampler._window = original_window


def merged(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((lo, hi))
    return result


cover = {k: merged(v) for k, v in seen.items()}
training_window_counts = {k: len(v) for k, v in seen.items()}
seen = {"prose": [], "code": []}
MixedSampler._window = tracked_window
gen_tracked = general_eval_set(base.prose, base.code, 256)
MixedSampler._window = original_window
assert gen_tracked == gen
eval_iterators = {k: iter(v) for k, v in seen.items()}
overlaps = []
fully_covered = 0
for s in gen:
    lo, hi = next(eval_iterators[s.kind])
    overlap = sum(max(0, min(hi, y) - max(lo, x)) for x, y in cover[s.kind])
    overlaps.append(overlap)
    fully_covered += overlap == 256
splits = split_eval_set(256)
record("train_eval_overlap", {
    "prose_chars": len(base.prose), "code_chars": len(base.code),
    "fusion_training_general_windows": training_window_counts,
    "eval_windows": len(gen), "eval_windows_any_training_overlap": sum(n > 0 for n in overlaps),
    "eval_windows_all_chars_seen_in_training": fully_covered,
    "eval_character_coverage": sum(overlaps) / (256 * len(gen)),
    "exact_split_view_A_train_eval_duplicates": sum(s.view_a in recall_train for s in splits),
    "fusion_training_distinct_split_view_A_windows": len(recall_train),
    "split_eval_A_digit_count": sum(len(s.a_blind) for s in splits),
    "split_eval_B_digit_count": sum(len(s.b_blind) for s in splits),
    "split_eval_A_digit_histogram": [sum(s.text[c] == str(d) for s in splits for c in s.a_blind) for d in range(10)],
    "split_eval_B_digit_histogram": [sum(s.text[c] == str(d) for s in splits for c in s.b_blind) for d in range(10)],
})

# Use headline architecture to count link parameters without training any model.
counts = {}
for signal in ("experts", "resid"):
    pair = FusedPair(setup.build_model(cfg.model_a, 97, cfg), setup.build_model(cfg.model_b, 1024, cfg),
                     [0, 1, 2, 3], 64, 4, signal, True, True)
    counts[signal] = sum(p.numel() for p in pair.bridge_parameters())
record("bridge_parameter_counts", counts)

# Exercise the real CLI with a deliberately leaky tiny checkpoint: does it fail?
cli_dir = OUT / "leaky_cli"
cli_dir.mkdir(exist_ok=True)
cli_cfg = {"out_dir": str(cli_dir), "seq_chars": 128,
           "model_a": {"tokenizer": "char"},
           "model_b": {"tokenizer": "bpe", "bpe_vocab": 220},
           "fused": {"d_bus": 8, "n_heads": 2,
                     "variants": {"deliberately_leaky": {"pair": "ab", "links": [0, 1],
                                                          "signal": "experts", "rel_pos": True}}}}
(cli_dir / "config.yaml").write_text(yaml.safe_dump(cli_cfg))
tb.save(cli_dir / "bpe220.json")
A, B = tiny(97, length=128), tiny(tb.vocab_size, 12, 1, length=128)
save(A, cli_dir / "model_a.pt")
save(B, cli_dir / "model_b.pt")
pair = FusedPair(A, B, [0, 1], 8, 2, "experts", True, False)
with torch.no_grad():
    for lk in pair.links.values():
        lk.rx_a.gate.fill_(1.)
        lk.rx_b.gate.fill_(1.)
torch.save(pair.links.state_dict(), cli_dir / "fused_deliberately_leaky.pt")
proc = subprocess.run([sys.executable, "scripts/toy_fused_audit.py", str(cli_dir / "config.yaml"),
                       "2", "1", "deliberately_leaky"], capture_output=True, text=True,
                      env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "."}, timeout=180)
(cli_dir / "stdout.txt").write_text(proc.stdout)
(cli_dir / "stderr.txt").write_text(proc.stderr)
cli_result = {"exit_code": proc.returncode, "stdout": proc.stdout}
if (cli_dir / "fused_audit.json").exists():
    cli_result["report"] = json.loads((cli_dir / "fused_audit.json").read_text())
    doc = general_eval_set(base.prose, base.code, 128, n=1)[0]
    a, b = make_batch([doc], ta, 128), make_batch([doc], tb, 128)
    pos = list(range(127))
    with torch.no_grad():
        full, _ = run_pair(pair, a, b)
        honest = prefix_logits(pair, doc.text, doc.text, pos)
        nll_full = -full[0].log_softmax(-1).gather(1, a.targets[0, :, None])[:, 0]
        nll_honest = -honest.log_softmax(-1).gather(1, a.targets[0, :, None])[:, 0]
    cli_result["max_logit_delta_same_doc"] = float((full[0] - honest).abs().max())
    cli_result["max_gold_nll_delta_same_doc"] = float((nll_full - nll_honest).abs().max())
record("leaky_audit_cli", cli_result)

(OUT / "evidence.json").write_text(json.dumps(REPORT, indent=2))
