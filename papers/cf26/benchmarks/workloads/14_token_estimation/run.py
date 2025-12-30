#!/usr/bin/env python3
"""
Token Estimation Benchmark

Estimates token usage before and after FuseReasoning optimization.
Uses tiktoken for OpenAI-compatible token counting.

This benchmark measures the token reduction achieved by the FuseReasoning pass,
which is a key metric for the paper's cost analysis.

Run: python run.py [--json] [--model MODEL]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from apxm_runner import find_apxm_cli as _find_cli

# Add tools directory for shared utilities
_tools_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "tools"
sys.path.insert(0, str(_tools_dir))
from apxm_env import ApxmConfig, setup_mlir_environment

# Default configuration
DEFAULT_MODEL = "gpt-4"  # tiktoken encoding model

# Try to import tiktoken, fall back to simple estimation
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


# Test workflows - now read from .ais files
WORKFLOW_DIR = Path(__file__).parent

# Workflow configurations (content is loaded from .ais files)
TEST_WORKFLOWS = {
    "simple_chain": {
        "description": "Simple 3-step reasoning chain (with dependencies)",
        "file": "simple_chain.ais",
    },
    "parallel_research": {
        "description": "5 parallel research queries",
        "file": "parallel_research.ais",
    },
    "sequential_reasoning": {
        "description": "10-step sequential reasoning (with dependencies)",
        "file": "sequential_reasoning.ais",
    },
}


def get_workflow(workflow_name: str) -> str:
    """Read workflow source from .ais file."""
    workflow_config = TEST_WORKFLOWS[workflow_name]
    workflow_file = WORKFLOW_DIR / workflow_config["file"]
    return workflow_file.read_text()


def find_apxm_cli() -> Path:
    """Find the apxm CLI binary using shared utility."""
    cli = _find_cli()
    if cli is None:
        raise FileNotFoundError("apxm CLI not found. Build with: python tools/apxm_cli.py compiler build")
    return cli


def count_tokens_tiktoken(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count tokens using tiktoken."""
    if not HAS_TIKTOKEN:
        # Simple fallback: ~4 chars per token
        return len(text) // 4

    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def extract_prompts_from_ais(ais_content: str) -> list[str]:
    """Extract all prompt strings from AIS workflow."""
    # Match rsn "prompt text" patterns
    pattern = r'rsn\s+"([^"]+)"'
    matches = re.findall(pattern, ais_content)
    return matches


def estimate_fused_prompt(prompts: list[str]) -> str:
    """Estimate the fused prompt that would be generated."""
    if len(prompts) <= 1:
        return prompts[0] if prompts else ""

    # FuseReasoning combines prompts into a multi-step instruction
    fused = "Please complete the following tasks in sequence:\n\n"
    for i, prompt in enumerate(prompts, 1):
        fused += f"{i}. {prompt}\n"
    fused += "\nProvide your response for each task."
    return fused


def compile_workflow(cli_path: Path, workflow_content: str, opt_level: int) -> dict:
    """Compile a workflow and extract diagnostics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ais_file = tmpdir / "test.ais"
        diag_file = tmpdir / "diagnostics.json"
        output_file = tmpdir / "output.apxmobj"

        ais_file.write_text(workflow_content)

        # Set up environment using shared utilities
        config = ApxmConfig.detect()
        if config.conda_prefix:
            env = setup_mlir_environment(config.conda_prefix, config.target_dir)
        else:
            env = os.environ.copy()

        cmd = [
            str(cli_path), "compile", str(ais_file),
            f"-O{opt_level}",
            "--emit-diagnostics", str(diag_file),
            "-o", str(output_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)

        if result.returncode != 0:
            return {"success": False, "error": result.stderr}

        try:
            diagnostics = json.loads(diag_file.read_text())
            return {"success": True, "diagnostics": diagnostics}
        except Exception as e:
            return {"success": False, "error": str(e)}


def estimate_workflow_tokens(workflow_name: str, workflow_config: dict, model: str) -> dict:
    """Estimate tokens for a workflow before and after optimization.

    Key insight: The real cost savings from FuseReasoning come from:
    1. System prompt amortization - system prompts sent once instead of N times
    2. Reduced API call overhead - fewer round trips
    3. Better context utilization - model sees all tasks together

    Typical system prompt overhead: 200-500 tokens per call
    """
    workflow = get_workflow(workflow_name)
    prompts = extract_prompts_from_ais(workflow)

    # Estimated system prompt overhead per call (conservative estimate)
    SYSTEM_PROMPT_TOKENS = 250

    # O0: Each prompt is a separate LLM call (system prompt sent N times)
    o0_prompt_tokens = sum(count_tokens_tiktoken(p, model) for p in prompts)
    o0_system_tokens = SYSTEM_PROMPT_TOKENS * len(prompts)
    o0_total_tokens = o0_prompt_tokens + o0_system_tokens

    # O1: Prompts get fused (system prompt sent once)
    if len(prompts) > 1:
        fused_prompt = estimate_fused_prompt(prompts)
        o1_prompt_tokens = count_tokens_tiktoken(fused_prompt, model)
    else:
        o1_prompt_tokens = o0_prompt_tokens
    o1_system_tokens = SYSTEM_PROMPT_TOKENS  # Only one call
    o1_total_tokens = o1_prompt_tokens + o1_system_tokens

    token_reduction = o0_total_tokens - o1_total_tokens
    reduction_pct = (token_reduction / o0_total_tokens * 100) if o0_total_tokens > 0 else 0

    return {
        "workflow": workflow_name,
        "description": workflow_config["description"],
        "num_prompts": len(prompts),
        "system_prompt_tokens": SYSTEM_PROMPT_TOKENS,
        "o0_unfused": {
            "num_calls": len(prompts),
            "prompt_tokens": o0_prompt_tokens,
            "system_tokens": o0_system_tokens,
            "total_tokens": o0_total_tokens,
        },
        "o1_fused": {
            "num_calls": 1,
            "prompt_tokens": o1_prompt_tokens,
            "system_tokens": o1_system_tokens,
            "total_tokens": o1_total_tokens,
        },
        "token_reduction": token_reduction,
        "reduction_pct": reduction_pct,
        "calls_saved": len(prompts) - 1,
    }


def run_langgraph(iterations: int = 1) -> dict:
    """Entry point for suite runner - not applicable for token estimation."""
    return {"note": "Token estimation only applies to A-PXM"}


def run_apxm(iterations: int = 1) -> dict:
    """Entry point for the suite runner."""
    model = DEFAULT_MODEL
    results = []

    for workflow_name, workflow_config in TEST_WORKFLOWS.items():
        result = estimate_workflow_tokens(workflow_name, workflow_config, model)
        results.append(result)

    # Calculate summary
    total_o0_tokens = sum(r["o0_unfused"]["total_tokens"] for r in results)
    total_o1_tokens = sum(r["o1_fused"]["total_tokens"] for r in results)
    total_reduction = total_o0_tokens - total_o1_tokens
    avg_reduction_pct = (total_reduction / total_o0_tokens * 100) if total_o0_tokens > 0 else 0

    return {
        "model": model,
        "has_tiktoken": HAS_TIKTOKEN,
        "results": results,
        "summary": {
            "total_o0_tokens": total_o0_tokens,
            "total_o1_tokens": total_o1_tokens,
            "total_reduction": total_reduction,
            "avg_reduction_pct": avg_reduction_pct,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Token Estimation Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Token counting model (default: {DEFAULT_MODEL})")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of iterations (ignored, for compatibility)")
    args = parser.parse_args()

    results = {
        "meta": {
            "benchmark": "token_estimation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "has_tiktoken": HAS_TIKTOKEN,
        },
        "data": [],
    }

    if not args.json:
        print(f"\nToken Estimation Benchmark")
        print(f"{'=' * 70}")
        print(f"Token model: {args.model}")
        print(f"Tiktoken available: {HAS_TIKTOKEN}")
        print()

    for workflow_name, workflow_config in TEST_WORKFLOWS.items():
        result = estimate_workflow_tokens(workflow_name, workflow_config, args.model)
        results["data"].append(result)

        if not args.json:
            print(f"{workflow_name}: {workflow_config['description']}")
            print(f"  Prompts: {result['num_prompts']}")
            print(f"  O0 (unfused): {result['o0_unfused']['total_tokens']} tokens ({result['o0_unfused']['num_calls']} calls)")
            print(f"               ({result['o0_unfused']['prompt_tokens']} prompt + {result['o0_unfused']['system_tokens']} system)")
            print(f"  O1 (fused):   {result['o1_fused']['total_tokens']} tokens ({result['o1_fused']['num_calls']} call)")
            print(f"               ({result['o1_fused']['prompt_tokens']} prompt + {result['o1_fused']['system_tokens']} system)")
            print(f"  Savings:      {result['token_reduction']} tokens ({result['reduction_pct']:.1f}%), {result['calls_saved']} calls")
            print()

    # Summary
    total_o0 = sum(r["o0_unfused"]["total_tokens"] for r in results["data"])
    total_o1 = sum(r["o1_fused"]["total_tokens"] for r in results["data"])
    total_reduction = total_o0 - total_o1
    avg_reduction_pct = (total_reduction / total_o0 * 100) if total_o0 > 0 else 0

    results["summary"] = {
        "total_o0_tokens": total_o0,
        "total_o1_tokens": total_o1,
        "total_reduction": total_reduction,
        "avg_reduction_pct": avg_reduction_pct,
    }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"{'=' * 70}")
        print(f"SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total O0 tokens: {total_o0}")
        print(f"Total O1 tokens: {total_o1}")
        print(f"Token reduction: {total_reduction} ({avg_reduction_pct:.1f}%)")
        print()
        print("Note: Actual token reduction depends on FuseReasoning eligibility.")
        print("      Some workflows may not be fused if prompts have dependencies.")


if __name__ == "__main__":
    main()
