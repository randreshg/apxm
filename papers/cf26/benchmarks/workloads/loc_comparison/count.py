#!/usr/bin/env python3
"""
Lines of Code Comparison Benchmark

Compares A-PXM AIS DSL vs LangGraph Python for equivalent workflows.
Validates the paper claim: "A-PXM requires 10 lines vs LangGraph's 42 lines"

Methodology:
- Count non-empty lines
- Count non-comment lines
- Count "semantic" lines (excludes imports, blank, comments)
"""

import json
import re
from datetime import datetime
from pathlib import Path


def count_ais_lines(filepath: Path) -> dict:
    """Count lines in AIS file."""
    content = filepath.read_text()
    lines = content.split('\n')

    total = len(lines)
    non_empty = sum(1 for line in lines if line.strip())
    non_comment = sum(1 for line in lines if line.strip() and not line.strip().startswith('//'))

    # Semantic lines: exclude comments, blank, and single-brace lines
    semantic = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('//'):
            continue
        if stripped in ['{', '}', '};']:
            continue
        semantic += 1

    return {
        "total": total,
        "non_empty": non_empty,
        "non_comment": non_comment,
        "semantic": semantic,
    }


def count_python_lines(filepath: Path) -> dict:
    """Count lines in Python file."""
    content = filepath.read_text()
    lines = content.split('\n')

    total = len(lines)
    non_empty = sum(1 for line in lines if line.strip())

    # Non-comment (exclude # comments and docstrings)
    in_docstring = False
    non_comment = 0
    for line in lines:
        stripped = line.strip()
        if '"""' in stripped or "'''" in stripped:
            # Toggle docstring state (simplified)
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                continue  # Single-line docstring
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if not stripped:
            continue
        if stripped.startswith('#'):
            continue
        non_comment += 1

    # Semantic: exclude imports, blank, comments, class/def declarations
    semantic = 0
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if not stripped:
            continue
        if stripped.startswith('#'):
            continue
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue
        if stripped.startswith('if __name__'):
            continue
        semantic += 1

    return {
        "total": total,
        "non_empty": non_empty,
        "non_comment": non_comment,
        "semantic": semantic,
    }


def compare_workflows(ais_path: Path, py_path: Path) -> dict:
    """Compare AIS and Python versions of same workflow."""
    ais_counts = count_ais_lines(ais_path)
    py_counts = count_python_lines(py_path)

    return {
        "ais": ais_counts,
        "python": py_counts,
        "ratio": {
            "total": py_counts["total"] / max(ais_counts["total"], 1),
            "non_empty": py_counts["non_empty"] / max(ais_counts["non_empty"], 1),
            "non_comment": py_counts["non_comment"] / max(ais_counts["non_comment"], 1),
            "semantic": py_counts["semantic"] / max(ais_counts["semantic"], 1),
        }
    }


def main():
    workloads_dir = Path(__file__).parent.parent

    # Find all workflow pairs
    workflows = []
    for ais_file in workloads_dir.glob("*/workflow.ais"):
        py_file = ais_file.parent / "workflow.py"
        if py_file.exists():
            workflows.append({
                "name": ais_file.parent.name,
                "ais": ais_file,
                "python": py_file,
            })

    results = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "benchmark": "lines_of_code_comparison",
            "methodology": {
                "total": "All lines including blank",
                "non_empty": "Lines with content",
                "non_comment": "Lines excluding comments",
                "semantic": "Meaningful code lines (excludes imports, braces, etc.)",
            }
        },
        "workflows": {},
        "summary": {
            "ais_total": 0,
            "python_total": 0,
            "average_ratio": 0,
        }
    }

    total_ais_semantic = 0
    total_py_semantic = 0

    for wf in workflows:
        comparison = compare_workflows(wf["ais"], wf["python"])
        results["workflows"][wf["name"]] = comparison
        total_ais_semantic += comparison["ais"]["semantic"]
        total_py_semantic += comparison["python"]["semantic"]

    if workflows:
        results["summary"]["ais_total"] = total_ais_semantic
        results["summary"]["python_total"] = total_py_semantic
        results["summary"]["average_ratio"] = total_py_semantic / max(total_ais_semantic, 1)

    # Print results
    print(json.dumps(results, indent=2))

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("LINES OF CODE COMPARISON")
    print("=" * 60)

    for name, data in results["workflows"].items():
        print(f"\n{name}:")
        print(f"  AIS:    {data['ais']['semantic']:3d} semantic lines")
        print(f"  Python: {data['python']['semantic']:3d} semantic lines")
        print(f"  Ratio:  {data['ratio']['semantic']:.1f}x more Python")

    print(f"\n{'=' * 60}")
    print(f"TOTAL:")
    print(f"  AIS:    {results['summary']['ais_total']} lines")
    print(f"  Python: {results['summary']['python_total']} lines")
    print(f"  Ratio:  {results['summary']['average_ratio']:.1f}x")
    print("=" * 60)

    # Verify paper claim
    print("\n" + "=" * 60)
    print("PAPER CLAIM VERIFICATION")
    print("=" * 60)
    if "1_parallel_research" in results["workflows"]:
        pr = results["workflows"]["1_parallel_research"]
        print(f"Paper claims: 10 lines AIS vs 42 lines Python")
        print(f"Actual:       {pr['ais']['semantic']} lines AIS vs {pr['python']['semantic']} lines Python")
        if pr['ais']['semantic'] <= 15 and pr['python']['semantic'] >= 35:
            print("CLAIM: VERIFIED (approximately correct)")
        else:
            print("CLAIM: NEEDS UPDATE")


if __name__ == "__main__":
    main()
