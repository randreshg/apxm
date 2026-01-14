#!/usr/bin/env python3
"""
Dataset Evaluation Utilities

Provides metrics (F1, Exact Match, accuracy) and dataset loaders
for dataset-driven benchmarks. Adapted from testing/LLMCompiler.
"""

import json
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (from LLMCompiler evaluation_utils)."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    if s is None:
        return ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_number(s: str) -> bool:
    """Check if string represents a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def compare_answer(answer: str, label: str) -> bool:
    """
    Compare answer and label (from LLMCompiler evaluation_utils).
    
    - If label is numeric, allows 10% margin
    - Otherwise does normalized string matching
    """
    if answer is None:
        return False

    # Check if label is numeric
    if is_number(label):
        label_val = float(label)
        try:
            answer_val = float(answer)
        except (ValueError, TypeError):
            return False
        # Allow 10% margin
        if answer_val > label_val * 0.9 and answer_val < label_val * 1.1:
            return True
        else:
            return False
    else:
        # Normalized string comparison
        label_norm = normalize_answer(label)
        answer_norm = normalize_answer(answer)
        return answer_norm == label_norm


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score between prediction and ground truth.
    
    Adapted from HotpotQA evaluation: token-level overlap.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 1.0 if pred_tokens == truth_tokens else 0.0
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Compute exact match (normalized string equality)."""
    return compare_answer(prediction, ground_truth)


def evaluate_dataset_results(
    results: Dict[str, Dict[str, Any]],
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate dataset results and compute aggregate metrics.
    
    Args:
        results: Dictionary mapping example ID to result dict with keys:
            - "answer": predicted answer (str)
            - "label": ground truth answer (str)
            - "question": question text (optional)
            - "time": execution time (optional)
        max_samples: Maximum number of samples to evaluate (None = all)
    
    Returns:
        Dictionary with metrics:
            - "accuracy": fraction of exact matches
            - "f1_mean": mean F1 score
            - "f1_std": standard deviation of F1 scores
            - "num_samples": number of evaluated samples
            - "latency_mean": mean latency (if available)
            - "latency_std": std latency (if available)
    """
    import statistics
    
    samples = list(results.items())
    if max_samples:
        samples = samples[:max_samples]
    
    exact_matches = []
    f1_scores = []
    latencies = []
    
    for example_id, result in samples:
        answer = result.get("answer", "")
        label = result.get("label", "")
        
        if not answer or not label:
            continue
        
        # Exact match
        em = compute_exact_match(answer, label)
        exact_matches.append(1.0 if em else 0.0)
        
        # F1 score
        f1 = compute_f1_score(answer, label)
        f1_scores.append(f1)
        
        # Latency (if available)
        if "time" in result:
            try:
                latencies.append(float(result["time"]))
            except (ValueError, TypeError):
                pass
    
    metrics = {
        "num_samples": len(exact_matches),
        "accuracy": statistics.mean(exact_matches) if exact_matches else 0.0,
    }
    
    if f1_scores:
        metrics["f1_mean"] = statistics.mean(f1_scores)
        metrics["f1_std"] = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0
        metrics["f1_min"] = min(f1_scores)
        metrics["f1_max"] = max(f1_scores)
    
    if latencies:
        metrics["latency_mean"] = statistics.mean(latencies)
        metrics["latency_std"] = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    
    return metrics


def load_hotpotqa_dataset(dataset_path: Optional[Path] = None, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load HotpotQA dataset.
    
    Args:
        dataset_path: Path to hotpotqa_comparison.json (defaults to papers/cf26/benchmarks/datasets/)
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        List of examples, each with keys: id, question, answer
    """
    if dataset_path is None:
        # Default to benchmarks/datasets location
        benchmarks_dir = Path(__file__).parent.parent
        dataset_path = benchmarks_dir / "datasets" / "hotpotqa_comparison.json"
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data


def load_parallelqa_dataset(dataset_path: Optional[Path] = None, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load ParallelQA dataset.
    
    Args:
        dataset_path: Path to parallelqa_dataset.json (defaults to papers/cf26/benchmarks/datasets/)
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        List of examples, each with keys: id, question, answer, branch
    """
    if dataset_path is None:
        # Default to benchmarks/datasets location
        benchmarks_dir = Path(__file__).parent.parent
        dataset_path = benchmarks_dir / "datasets" / "parallelqa_dataset.json"
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data


def load_movie_dataset(dataset_path: Optional[Path] = None, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load Movie Recommendations dataset.
    
    Args:
        dataset_path: Path to movie_recommendations_formatted.json (defaults to papers/cf26/benchmarks/datasets/)
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        List of examples, each with keys: id, question, answer
    """
    if dataset_path is None:
        # Default to benchmarks/datasets location
        benchmarks_dir = Path(__file__).parent.parent
        dataset_path = benchmarks_dir / "datasets" / "movie_recommendations_formatted.json"
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data
