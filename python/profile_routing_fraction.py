#!/usr/bin/env python3
"""
profile_routing_fraction.py — Measure what % of inference time the MoE routing gate takes.

Hooks into OLMoE-1B-7B and profiles:
  - Time in routing gates (OlmoeTopKRouter — the component we replace)
  - Time in expert MLPs (OlmoeExperts)
  - Time in attention layers
  - Time in everything else

Usage:
    python3 python/profile_routing_fraction.py --model-dir /path/to/olmoe-1b-7b
"""

import argparse
import time
import sys
import os
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def profile_model(model_dir: str, max_tokens: int = 512, n_runs: int = 5,
                  device: str = "cuda"):
    """Profile OLMoE forward pass, breaking down time by component."""

    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device: {device}")

    # -- Discover modules by TYPE, not by name --
    gate_modules = []
    expert_modules = []
    moe_blocks = []
    attn_modules = []

    for name, module in model.named_modules():
        mod_type = type(module).__name__

        if mod_type == "OlmoeTopKRouter":
            gate_modules.append((name, module))
        elif mod_type == "OlmoeExperts":
            expert_modules.append((name, module))
        elif mod_type == "OlmoeSparseMoeBlock":
            moe_blocks.append((name, module))
        elif "self_attn" in name and mod_type == "OlmoeAttention":
            attn_modules.append((name, module))

    # Fallback: if OLMoE-specific types not found, try generic
    if not gate_modules:
        for name, module in model.named_modules():
            mod_type = type(module).__name__
            if "router" in mod_type.lower() or "router" in name.lower():
                gate_modules.append((name, module))
            elif "expert" in mod_type.lower() and "expert" not in name.split(".")[-2:-1]:
                expert_modules.append((name, module))

    # Fallback for attention
    if not attn_modules:
        for name, module in model.named_modules():
            mod_type = type(module).__name__
            if "attention" in mod_type.lower() or "attn" in mod_type.lower():
                has_children = any(True for _ in module.children())
                if has_children:
                    attn_modules.append((name, module))

    print(f"\nFound {len(gate_modules)} gate/router modules (OlmoeTopKRouter)")
    print(f"Found {len(expert_modules)} expert modules (OlmoeExperts)")
    print(f"Found {len(moe_blocks)} MoE blocks (OlmoeSparseMoeBlock)")
    print(f"Found {len(attn_modules)} attention modules")

    if gate_modules:
        print(f"  Example gate: {gate_modules[0][0]} ({type(gate_modules[0][1]).__name__})")
    if expert_modules:
        print(f"  Example experts: {expert_modules[0][0]} ({type(expert_modules[0][1]).__name__})")

    if not gate_modules:
        print("\nERROR: Could not find gate modules!")
        print("Dumping all module types:")
        seen = set()
        for name, module in model.named_modules():
            t = type(module).__name__
            if t not in seen:
                seen.add(t)
                print(f"  {t}: {name}")
        return

    # -- Prepare input --
    text = "The quick brown fox jumps over the lazy dog. " * 30
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_tokens)
    input_ids = inputs.input_ids.to(device)
    print(f"\nInput shape: {input_ids.shape}")

    # -- Hook-based profiling --
    timings = defaultdict(float)
    call_counts = defaultdict(int)
    start_times = {}

    def make_pre_hook(category, idx):
        def pre_hook(module, input):
            torch.cuda.synchronize()
            start_times[f"{category}_{idx}"] = time.perf_counter()
        return pre_hook

    def make_post_hook(category, idx):
        def post_hook(module, input, output):
            torch.cuda.synchronize()
            key = f"{category}_{idx}"
            if key in start_times:
                elapsed = time.perf_counter() - start_times[key]
                timings[category] += elapsed
                call_counts[category] += 1
        return post_hook

    handles = []

    for i, (name, mod) in enumerate(gate_modules):
        handles.append(mod.register_forward_pre_hook(make_pre_hook("routing_gate", i)))
        handles.append(mod.register_forward_hook(make_post_hook("routing_gate", i)))

    for i, (name, mod) in enumerate(expert_modules):
        handles.append(mod.register_forward_pre_hook(make_pre_hook("expert_mlps", i)))
        handles.append(mod.register_forward_hook(make_post_hook("expert_mlps", i)))

    for i, (name, mod) in enumerate(moe_blocks):
        handles.append(mod.register_forward_pre_hook(make_pre_hook("moe_block_total", i)))
        handles.append(mod.register_forward_hook(make_post_hook("moe_block_total", i)))

    for i, (name, mod) in enumerate(attn_modules):
        handles.append(mod.register_forward_pre_hook(make_pre_hook("attention", i)))
        handles.append(mod.register_forward_hook(make_post_hook("attention", i)))

    # -- Warmup --
    print("Warming up (3 runs)...")
    with torch.no_grad():
        for _ in range(3):
            model(input_ids)
    torch.cuda.synchronize()

    timings.clear()
    call_counts.clear()
    start_times.clear()

    # -- Profile --
    print(f"Profiling ({n_runs} runs)...")
    total_time = 0.0

    with torch.no_grad():
        for run in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_time += (t1 - t0)

    for h in handles:
        h.remove()

    # -- Results --
    avg_total_ms = (total_time / n_runs) * 1000

    print(f"\n{'='*65}")
    print(f"PROFILING RESULTS ({n_runs} runs, {input_ids.shape[1]} tokens)")
    print(f"{'='*65}")
    print(f"Total forward pass:     {avg_total_ms:.2f} ms")
    print()

    for category in ["routing_gate", "expert_mlps", "moe_block_total", "attention"]:
        if category in timings:
            avg_ms = (timings[category] / n_runs) * 1000
            pct = (timings[category] / total_time) * 100
            avg_calls = call_counts[category] / n_runs
            print(f"  {category:25s}: {avg_ms:8.2f} ms  ({pct:5.2f}%)  "
                  f"[{avg_calls:.0f} calls/fwd]")

    measured = sum(timings.values())
    other_time = total_time - measured
    other_pct = (other_time / total_time) * 100
    other_ms = (other_time / n_runs) * 1000
    print(f"  {'other (embed, norm, etc)':25s}: {other_ms:8.2f} ms  ({other_pct:5.2f}%)")

    print()
    if "routing_gate" in timings and "expert_mlps" in timings:
        gate_ms = (timings["routing_gate"] / n_runs) * 1000
        gate_pct = (timings["routing_gate"] / total_time) * 100
        expert_ms = (timings["expert_mlps"] / n_runs) * 1000
        ratio = gate_ms / expert_ms * 100 if expert_ms > 0 else 0

        print(f"  >>> ROUTING = {gate_pct:.2f}% of total inference time ({gate_ms:.3f} ms)")
        print(f"  >>> EXPERTS = {(timings['expert_mlps']/total_time*100):.2f}% of total ({expert_ms:.2f} ms)")
        print(f"  >>> ROUTING is {ratio:.1f}% of expert compute time")
        print(f"  >>> (SpectralAI replaces the routing gate with RT Core BVH)")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile routing gate fraction in OLMoE inference")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to OLMoE-1B-7B model")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Input sequence length")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of profiling runs")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    profile_model(args.model_dir, args.max_tokens, args.n_runs, args.device)
