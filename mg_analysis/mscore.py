import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import coloredlogs
import numpy as np
import pandas as pd
from plot import (
    create_pareto_plot,
    visualize_bias_rate,
    visualize_classes,
    visualize_marker_types,
    visualize_mg_count,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


@dataclass
class ParetoVector:
    """Represents a model's position in Pareto space."""

    MG_rate: float
    INCL_rate: float
    NEUT_rate: float


@dataclass
class MScoreResult:
    """M-score results for a dataset."""

    dataset: str
    content_type: Optional[str]
    # overall_m_score: float
    # average_m_score: float
    bias_rate: float
    total_human_nouns: int
    total_masc_gen_nouns: int
    total_neut: int
    total_incl: int
    total_incl_greetings: int
    total_neutral_prons: int
    pareto_vector: ParetoVector
    detailed_scores: list[dict]


MODEL_TYPE_NAMES = [
    ("human", "oracle"),
    ("human", "oasst2"),
    ("llm_prop", "gemini"),
    ("llm_prop", "gpt4o_mini"),
    ("llm_prop", "claude-3-haiku"),
    ("llm_local", "llama"),
    ("llm_local", "ministral"),
    ("llm_local", "mistral-small"),
]

# SHELVED
DEFAULT_WEIGHTS = (1.925, -1.839, -1.027, -0.501, -1.526)  # mg, i, g, n, p


def mscore_to_dict(mscore: MScoreResult) -> dict:
    """Convert MScoreResult dataclass to dict for plotting functions."""
    result = asdict(mscore)
    # Also convert nested ParetoVector dataclass to dict
    result["pareto_vector"] = asdict(mscore.pareto_vector)
    return result


def mscores_to_dicts(mscores: list[MScoreResult]) -> list[dict]:
    """Convert list of MScoreResult to list of dicts."""
    return [mscore_to_dict(m) for m in mscores]


def load_results(file_path: str) -> list[dict]:
    """Load analysis results from a JSON file."""
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
        logger.info(f"Loaded {len(data)} results from {file_path}")
        return data


def merge_json_results(*file_paths: str) -> list[dict]:
    """Merge multiple JSON result files into a single list."""
    merged_results = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist. Skipping...")
            continue

        if not file_path.lower().endswith(".json"):
            logger.warning(f"File {file_path} is not a JSON file. Skipping...")
            continue

        try:
            results = load_results(file_path)
            merged_results.extend(results)
        except json.JSONDecodeError:
            logger.error(f"Could not parse JSON from {file_path}")
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")

    return merged_results


def save_dataframe(df: pd.DataFrame, filepath: str, formats: list[str] = ["csv"]):
    """Save a DataFrame in multiple formats."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "csv" in formats:
        df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf8")
        logger.info(f"Saved table to {path.with_suffix('.csv')}")

    if "latex" in formats:
        latex_path = path.with_suffix(".tex")
        with open(latex_path, "w", encoding="utf8") as f:
            f.write(df.to_latex(escape=True, index=False))
        logger.info(f"Saved LaTeX table to {latex_path}")


FP_IDX_E1 = [
    "llama_266",
    "llama_3610",
    "claude-3-haiku_6120",
    "claude-3-haiku_7216",
    "mistral-small_1429",
    "mistral-small_3132",
    "claude-3-haiku_1862",
    "gpt4o_mini_3936",
    "mistral-small_959",
]

FP_IDX_E2 = [
    "llama_431",
    "llama_432",
    "llama_788",
    "llama_1065",
    "llama_1329",
    "llama_1564",
    "llama_2007",
    "llama_2034",
    "llama_2417",
    "llama_3148",
    "llama_3352",
    "llama_3925",
    "llama_4020",
    "llama_4037",
    "llama_4117",
    "llama_4261",
    "ministral_887",
    "ministral_1328",
    "claude-3-haiku_1329",
    "gpt4o_mini_1329",
    "gpt4o_mini_3947",
    "gemini_1329",
    "mistral-small_4368",
]


def compute_marker_rates(
    results: list[dict], is_e2: bool = False
) -> tuple[float, float]:
    """
    Calculate the percentage of texts containing inclusive and neutral markers.
    """
    fp_idx = FP_IDX_E2 if is_e2 else FP_IDX_E1
    valid_results = [
        r for r in results if r and r.get("text_index_dataset") not in fp_idx
    ]
    total_texts = len(valid_results)

    if total_texts == 0:
        return 0.0, 0.0

    inclusive_count = 0
    neutral_count = 0

    for result in valid_results:
        # inclusive markers
        if (
            result.get("incl_greetings_logs")
            or result.get("neutral_prons_logs")
            or result.get("incl_pairs_logs")
            or result.get("separator_logs")
            or result.get("upper_logs")
        ):
            inclusive_count += 1

        # neutral markers
        if result.get("neutral_logs"):
            neutral_count += 1

    inclusive_rate = (inclusive_count / total_texts) * 100
    neutral_rate = (neutral_count / total_texts) * 100

    return inclusive_rate, neutral_rate


def bootstrap_marker_ci(
    results: list[dict],
    is_e2: bool = False,
    n_boot: int = 10000,
    confidence_level: float = 95.0,
) -> dict[str, tuple[float, float]]:
    """
    Bootstrap confidence intervals for marker rates.
    """
    rng = np.random.default_rng(42)
    boot_inclusive_rates = []
    boot_neutral_rates = []

    for _ in range(n_boot):
        sampled = rng.choice(results, size=len(results), replace=True)  # type: ignore[arg-type]
        inclusive_rate, neutral_rate = compute_marker_rates(sampled, is_e2=is_e2)
        boot_inclusive_rates.append(inclusive_rate)
        boot_neutral_rates.append(neutral_rate)

    # Compute percentiles for CI
    alpha = (100 - confidence_level) / 2
    lower_percentile = alpha
    upper_percentile = 100 - alpha

    inclusive_ci = (
        float(np.percentile(boot_inclusive_rates, lower_percentile)),
        float(np.percentile(boot_inclusive_rates, upper_percentile)),
    )

    neutral_ci = (
        float(np.percentile(boot_neutral_rates, lower_percentile)),
        float(np.percentile(boot_neutral_rates, upper_percentile)),
    )

    return {"inclusive": inclusive_ci, "neutral": neutral_ci}


def generate_marker_ci_csv(
    loaded_results_e1: list[list[dict]],
    loaded_results_e2: list[list[dict]],
    datasets: list[str],
    output_file: str = "analyses/marker_rates.csv",
    n_boot: int = 10000,
    confidence_level: float = 95.0,
) -> pd.DataFrame:
    """
    Generate CSV with marker rates and confidence intervals for both experiments.
    """
    logging.info(
        f"Generating marker rate CSV with {confidence_level}% confidence intervals..."
    )

    data = []

    for i, ds in enumerate(datasets):
        logging.info(f"Processing dataset: {ds}")
        row = {"dataset": ds}

        # Experiment 1
        res_e1 = loaded_results_e1[i]

        # Calculate point estimates
        incl_e1, neut_e1 = compute_marker_rates(res_e1, is_e2=False)

        # Calculate confidence intervals
        ci_dict_e1 = bootstrap_marker_ci(
            res_e1, is_e2=False, n_boot=n_boot, confidence_level=confidence_level
        )

        row["incl_e1"] = round(incl_e1, 2)  # type: ignore
        row["ci_incl_e1"] = (
            f"[{ci_dict_e1['inclusive'][0]:.2f}, {ci_dict_e1['inclusive'][1]:.2f}]"
        )
        row["neut_e1"] = round(neut_e1, 2)  # type: ignore
        row["ci_neut_e1"] = (
            f"[{ci_dict_e1['neutral'][0]:.2f}, {ci_dict_e1['neutral'][1]:.2f}]"
        )

        # Experiment 2 (skip for oracle and oasst2)
        if ds not in ["oracle", "oasst2"]:
            res_e2 = loaded_results_e2[i]

            # Calculate point estimates
            incl_e2, neut_e2 = compute_marker_rates(res_e2, is_e2=True)

            # Calculate confidence intervals
            ci_dict_e2 = bootstrap_marker_ci(
                res_e2, is_e2=True, n_boot=n_boot, confidence_level=confidence_level
            )

            row["incl_e2"] = round(incl_e2, 2)  # type: ignore
            row["ci_incl_e2"] = (
                f"[{ci_dict_e2['inclusive'][0]:.2f}, {ci_dict_e2['inclusive'][1]:.2f}]"
            )
            row["neut_e2"] = round(neut_e2, 2)  # type: ignore
            row["ci_neut_e2"] = (
                f"[{ci_dict_e2['neutral'][0]:.2f}, {ci_dict_e2['neutral'][1]:.2f}]"
            )
        else:
            # Leave empty for datasets not in Experiment 2
            row["incl_e2"] = ""
            row["ci_incl_e2"] = ""
            row["neut_e2"] = ""
            row["ci_neut_e2"] = ""

        data.append(row)

    df = pd.DataFrame(
        data,
        columns=[
            "dataset",
            "incl_e1",
            "ci_incl_e1",
            "neut_e1",
            "ci_neut_e1",
            "incl_e2",
            "ci_incl_e2",
            "neut_e2",
            "ci_neut_e2",
        ],
    )

    df.to_csv(output_file, index=False)
    logging.info(f"Saved marker rates with CIs to {output_file}")

    return df


# M-Score Calculation Functions
def apply_formula(
    *,
    masc_gen_count: int,
    human_noun_count: int,
    inclusive_count: int = 0,
    incl_greetings_count: int = 0,
    neutral_count: int = 0,
    neutral_prons_count: int = 0,
    weights: Optional[tuple[int, ...]] = None,
    clamp: bool = True,
    legacy: bool = False,
) -> float:
    """Compute M-score using either legacy or weighted formula."""
    if human_noun_count <= 0:
        return np.nan

    if legacy:
        return masc_gen_count / human_noun_count

    final_weights = weights or DEFAULT_WEIGHTS
    mg_w, i_w, g_w, n_w, p_w = final_weights

    score = (
        mg_w * masc_gen_count
        + i_w * inclusive_count
        + g_w * incl_greetings_count
        + n_w * neutral_count
        + p_w * neutral_prons_count
    ) / human_noun_count

    if clamp:
        return min(max(score, -3), 3)
    return score


def calculate_m_score(
    results: list[dict] | np.ndarray,
    dataset: str,
    content_type: Optional[str] = None,
    is_real: bool = True,
    legacy: bool = False,
    weights: Optional[tuple[int, ...]] = None,
) -> MScoreResult:
    """
    Calculate M-score-related metrics.
    Bias rate here is CONDITIONED ON texts with ≥1 human noun
    (same semantics as old code).
    """
    total_human_nouns = 0
    total_masc_gen_nouns = 0
    total_neut = 0
    total_incl = 0
    total_incl_greetings = 0
    total_neutral_prons = 0

    detailed_scores = []

    human_noun_count_label = "real_human_nouns_count" if is_real else "human_noun_count"
    masc_gen_logs_label = "real_masc_gen_logs" if is_real else "masc_gen_logs"

    for result in results:
        human_noun_count = result.get(human_noun_count_label, 0) or 0
        masc_gen_count = len(result.get(masc_gen_logs_label) or [])
        incl_greetings_count = len(result.get("incl_greetings_logs", []))
        incl_pairs_count = len(result.get("incl_pairs_logs", []))
        neutral_prons_count = len(result.get("neutral_prons_logs", []))
        separator_count = len(result.get("separator_logs", []))
        neutral_count = len(result.get("neutral_logs", []))

        inclusive_count = incl_pairs_count + separator_count

        if masc_gen_count > human_noun_count:
            masc_gen_count = human_noun_count

        total_human_nouns += human_noun_count
        total_masc_gen_nouns += masc_gen_count
        total_neut += neutral_count
        total_incl += inclusive_count
        total_incl_greetings += incl_greetings_count
        total_neutral_prons += neutral_prons_count

        has_valid_human_nouns = human_noun_count > 0
        is_text_biased = int(has_valid_human_nouns and masc_gen_count > 0)

        detailed_scores.append(
            {
                "text": result.get("text", ""),
                "human_noun_count": human_noun_count,
                "masc_gen_count": masc_gen_count,
                "neutral_count": neutral_count,
                "inclusive_count": inclusive_count,
                "has_valid_human_nouns": has_valid_human_nouns,
                "is_text_biased": is_text_biased,
            }
        )

    valid_texts = [x for x in detailed_scores if x["has_valid_human_nouns"]]

    if valid_texts:
        bias_rate_human_nouns = sum(x["is_text_biased"] for x in valid_texts) / len(
            valid_texts
        )
    else:
        bias_rate_human_nouns = 0.0

    pareto_vector = compute_pareto_vector(
        {
            "total_human_nouns": total_human_nouns,
            "total_masc_gen_nouns": total_masc_gen_nouns,
            "total_incl": total_incl,
            "total_incl_greetings": total_incl_greetings,
            "total_neutral_prons": total_neutral_prons,
            "total_neut": total_neut,
        }
    )

    return MScoreResult(
        dataset=dataset,
        content_type=content_type,
        bias_rate=bias_rate_human_nouns,
        total_human_nouns=total_human_nouns,
        total_masc_gen_nouns=total_masc_gen_nouns,
        total_neut=total_neut,
        total_incl=total_incl,
        total_incl_greetings=total_incl_greetings,
        total_neutral_prons=total_neutral_prons,
        pareto_vector=pareto_vector,
        detailed_scores=detailed_scores,
    )


def calculate_bias_rate_overall(
    results: list[dict] | np.ndarray,
    is_real: bool = True,
) -> float:
    masc_gen_logs_label = "real_masc_gen_logs" if is_real else "masc_gen_logs"
    total_texts = len(results)
    if total_texts == 0:
        return 0.0
    biased = sum(1 for r in results if r.get(masc_gen_logs_label))
    return biased / total_texts


def bootstrap_bias_rate_ci(
    results: list[dict],
    dataset: str,
    is_real: bool = True,
    n_boot: int = 100,
    rate_type: str = "overall",  # "overall" or "human_nouns"
) -> tuple[float, float]:
    """
    Bootstrap CI for bias rates
    """
    rng = np.random.default_rng(42)
    boot_rates = []

    for _ in range(n_boot):
        sampled = rng.choice(results, size=len(results), replace=True)  # type: ignore[arg-type]

        if rate_type == "overall":
            rate = calculate_bias_rate_overall(sampled, is_real=is_real)
        elif rate_type == "human_nouns":
            rate = calculate_m_score(sampled, dataset, is_real=is_real).bias_rate
        else:
            raise ValueError(f"Unknown rate_type: {rate_type}")

        boot_rates.append(rate)

    return (
        float(np.percentile(boot_rates, 2.5)),
        float(np.percentile(boot_rates, 97.5)),
    )


def generate_bias_rate_ci_csv(
    loaded_results_e1: list[list[dict]],
    loaded_results_e2: list[list[dict]],
    datasets: list[str],
    output_file: str = "analyses/bias_rates.csv",
    n_boot: int = 100,
) -> pd.DataFrame:
    """Generate CSV with bias rates and confidence intervals for both experiments."""
    logging.info("Generating bias rate CSV with confidence intervals...")

    data = []

    for i, ds in enumerate(datasets):
        logging.info(f"Processing dataset: {ds}")
        row = {"dataset": ds}

        # Exp 1
        res_e1 = loaded_results_e1[i]

        # MG ≥ 1 overall
        br_e1 = calculate_bias_rate_overall(res_e1, is_real=True)
        lower_ci_e1, upper_ci_e1 = bootstrap_bias_rate_ci(
            res_e1, ds, is_real=True, n_boot=n_boot, rate_type="overall"
        )
        row["br_e1"] = round(br_e1 * 100, 2)  # type: ignore
        row["ci_e1"] = f"[{lower_ci_e1 * 100:.3f}, {upper_ci_e1 * 100:.3f}]"

        # MG ≥ 1 with human nouns
        m_score_e1 = calculate_m_score(res_e1, ds, is_real=True)
        br_hn_e1 = m_score_e1.bias_rate
        lower_ci_hn_e1, upper_ci_hn_e1 = bootstrap_bias_rate_ci(
            res_e1, ds, is_real=True, n_boot=n_boot, rate_type="human_nouns"
        )
        row["br_hn_e1"] = round(br_hn_e1 * 100, 2)  # type: ignore
        row["ci_hn_e1"] = f"[{lower_ci_hn_e1 * 100:.3f}, {upper_ci_hn_e1 * 100:.3f}]"

        # Exp 2
        if ds not in ["oracle", "oasst2"]:
            res_e2 = loaded_results_e2[i]

            # MG ≥ 1 overall
            br_e2 = calculate_bias_rate_overall(res_e2, is_real=True)
            lower_ci_e2, upper_ci_e2 = bootstrap_bias_rate_ci(
                res_e2, ds, is_real=True, n_boot=n_boot, rate_type="overall"
            )
            row["br_e2"] = round(br_e2 * 100, 2)  # type: ignore
            row["ci_e2"] = f"[{lower_ci_e2 * 100:.3f}, {upper_ci_e2 * 100:.3f}]"

            # MG ≥ 1 with human nouns
            m_score_e2 = calculate_m_score(res_e2, ds, is_real=True)
            br_hn_e2 = m_score_e2.bias_rate
            lower_ci_hn_e2, upper_ci_hn_e2 = bootstrap_bias_rate_ci(
                res_e2, ds, is_real=True, n_boot=n_boot, rate_type="human_nouns"
            )
            row["br_hn_e2"] = round(br_hn_e2 * 100, 2)  # type: ignore
            row["ci_hn_e2"] = (
                f"[{lower_ci_hn_e2 * 100:.3f}, {upper_ci_hn_e2 * 100:.3f}]"
            )
        else:
            # leave empty for datasets not in Experiment 2
            row["br_e2"] = ""
            row["ci_e2"] = ""
            row["br_hn_e2"] = ""
            row["ci_hn_e2"] = ""

        data.append(row)

    df = pd.DataFrame(
        data,
        columns=[
            "dataset",
            "br_e1",
            "ci_e1",
            "br_e2",
            "ci_e2",
            "br_hn_e1",
            "ci_hn_e1",
            "br_hn_e2",
            "ci_hn_e2",
        ],
    )

    df.to_csv(output_file, index=False)
    logging.info(f"Saved bias rates with CIs to {output_file}")

    return df


# Pareto Analysis Functions
def compute_pareto_vector(m_score_results: dict) -> ParetoVector:
    """
    Compute Pareto bias vector from aggregated counts.
    Lower MG is better; higher INCL and NEUT are better.
    """
    hn = m_score_results["total_human_nouns"]

    if hn == 0:
        return ParetoVector(MG_rate=np.nan, INCL_rate=np.nan, NEUT_rate=np.nan)

    # merge I, G, and P into INCL
    INCL_rate_numerator = (
        m_score_results["total_incl"]
        + m_score_results["total_incl_greetings"]
        + m_score_results["total_neutral_prons"]
    )

    return ParetoVector(
        MG_rate=(m_score_results["total_masc_gen_nouns"] / hn) * 100,
        INCL_rate=(INCL_rate_numerator / hn) * 100,
        NEUT_rate=(m_score_results["total_neut"] / hn) * 100,
    )


def pareto_dominates(a: ParetoVector, b: ParetoVector) -> bool:
    """
    Returns True if model A Pareto-dominates model B.
    Lower MG is better; higher INCL and NEUT are better.
    We say that A Pareto-dominates B if:
    - A is better than or equal to B in all dimensions, AND
    - A is strictly better than B in at least one dimension.
    """
    # is A better or equal to B in all dimensions?
    better_or_equal = (
        a.MG_rate <= b.MG_rate
        and a.INCL_rate >= b.INCL_rate
        and a.NEUT_rate >= b.NEUT_rate
    )

    # is A strictly better than B in at least one dimension?
    strictly_better = (
        a.MG_rate < b.MG_rate or a.INCL_rate > b.INCL_rate or a.NEUT_rate > b.NEUT_rate
    )

    return better_or_equal and strictly_better


def compute_pareto_front(model_results: list[MScoreResult]) -> list[str]:
    """Identify datasets on the Pareto front."""
    # list of non-dominated models
    pareto_models = []

    for i, model_i in enumerate(model_results):
        dominated = False
        vec_i = model_i.pareto_vector

        for j, model_j in enumerate(model_results):
            if i == j:
                # same model, skip
                continue
            vec_j = model_j.pareto_vector

            if pareto_dominates(vec_j, vec_i):
                dominated = True
                break

        if not dominated:
            pareto_models.append(model_i.dataset)

    return pareto_models


def build_pareto_table(
    model_results: list[MScoreResult], pareto_front: list[str]
) -> list[dict]:
    """Build a table of Pareto analysis results."""
    table = []

    for m in model_results:
        vec = m.pareto_vector
        row = {
            "model": m.dataset,
            "MG/HN": round(vec.MG_rate, 4),
            "INCL/HN": round(vec.INCL_rate, 4),
            "NEUT/HN": round(vec.NEUT_rate, 4),
            "Pareto-optimal?": "✓" if m.dataset in pareto_front else "",
        }
        table.append(row)

    return table


def bootstrap_pareto_membership(
    loaded_results_by_model: dict[str, list[dict]],
    n_boot: int = 10000,
    seed: int = 42,
    is_e2: bool = False,
) -> dict[str, float]:
    """Bootstrap Pareto front stability analysis."""
    logger.info("Bootstrapping Pareto front membership...")
    rng = np.random.default_rng(seed)

    models = list(loaded_results_by_model.keys())
    pareto_counts = dict.fromkeys(models, 0)

    exp_label = "[E2]" if is_e2 else "[E1]"

    time_min = n_boot * 0.0005  # rough estimate: 0.5 ms per iteration
    logger.warning(
        f"{exp_label} Running Pareto with {n_boot} bootstrap iterations. This should take ~{time_min:.1f} minutes."
    )

    for i in range(n_boot):
        if i % (n_boot // 10) == 0:
            logger.info(f"{exp_label} Pareto bootstrap iteration {i + 1} / {n_boot}")

        boot_scores = []
        for model in models:
            results = loaded_results_by_model[model]
            sampled = rng.choice(results, size=len(results), replace=True)  # type: ignore[arg-type]
            m_score = calculate_m_score(sampled, model)
            boot_scores.append(m_score)

        pareto_front = compute_pareto_front(boot_scores)
        for m in pareto_front:
            pareto_counts[m] += 1

    pareto_probs = {m: pareto_counts[m] / n_boot for m in models}
    return pareto_probs


def run_pareto_analysis(
    m_scores_e1: list[MScoreResult],
    m_scores_e2: list[MScoreResult],
    loaded_by_model_e1: dict[str, list[dict]],
    loaded_by_model_e2: dict[str, list[dict]],
    ci: bool = False,
    n_boot: int = 10000,
    pdf: bool = False,
):
    """Run complete Pareto analysis for both experiments."""
    logger.info("Running Pareto analysis...")

    m_scores_e2_filtered = [
        m for m in m_scores_e2 if m.dataset not in ["oracle", "oasst2"]
    ]

    pareto_front_e1 = compute_pareto_front(m_scores_e1)
    pareto_front_e2 = compute_pareto_front(m_scores_e2_filtered)

    pareto_table_e1 = build_pareto_table(m_scores_e1, pareto_front_e1)
    pareto_table_e2 = build_pareto_table(m_scores_e2_filtered, pareto_front_e2)

    pareto_df_e1 = pd.DataFrame(pareto_table_e1)
    pareto_df_e2 = pd.DataFrame(pareto_table_e2)

    if ci:
        pareto_probs_e1 = bootstrap_pareto_membership(loaded_by_model_e1, n_boot=n_boot)

        pareto_probs_e2 = bootstrap_pareto_membership(
            loaded_by_model_e2, n_boot=n_boot, is_e2=True
        )

        pareto_df_e1["P(Pareto)"] = pareto_df_e1["model"].map(pareto_probs_e1)
        pareto_df_e2["P(Pareto)"] = pareto_df_e2["model"].map(pareto_probs_e2)

    # mere pareto_df_e1 and pareto_df_e2
    merged_pareto_df = pd.merge(
        pareto_df_e1,
        pareto_df_e2,
        on="model",
        how="outer",
        suffixes=("_e1", "_e2"),
    )

    save_dataframe(
        merged_pareto_df,
        filepath="analyses/pareto.csv",
        formats=["csv"],
    )

    m_scores_e1_dicts = mscores_to_dicts(m_scores_e1)
    m_scores_e2_dicts = mscores_to_dicts(m_scores_e2_filtered)

    create_pareto_plot(
        m_scores_e1_dicts,
        m_scores_e2_dicts,
        pareto_front_e1,
        pareto_front_e2,
        output_file="plots/pareto.svg",
        pdf=pdf,
    )
    logger.info("Saved Pareto plot")


# MG Count Analysis Functions
def get_mg_count(results: list[dict], filter_epicenes: bool = False) -> pd.DataFrame:
    """Extract masculine generic noun counts from results."""
    masc_gen_nouns = []
    dataset_name = results[0]["dataset"] if results else "unknown"

    for dataset_result in results:
        masc_gen_logs = dataset_result.get("real_masc_gen_logs", [])
        # masc_gen_nouns.extend([log["masc_gen"] for log in masc_gen_logs])
        for log in masc_gen_logs:
            if filter_epicenes and log["from_epicene"] == 1:
                continue
            masc_gen_nouns.append(log["masc_gen"])

    masc_gen_df = pd.DataFrame(masc_gen_nouns, columns=["noun"])
    masc_gen_df = masc_gen_df.value_counts().reset_index()
    masc_gen_df["dataset"] = dataset_name

    return masc_gen_df


def get_mg_total(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate MG counts across all datasets."""
    combined_df = pd.concat(dfs, ignore_index=True)
    ranked_nouns = combined_df.groupby("noun")["count"].sum().reset_index()
    ranked_nouns = ranked_nouns.sort_values("count", ascending=False).reset_index(
        drop=True
    )
    ranked_nouns["rank"] = ranked_nouns.index + 1

    cols = ["rank"] + [col for col in ranked_nouns.columns if col != "rank"]
    return ranked_nouns[cols]


# Data Loading
def load_all_results(
    datasets: list[str], results_e1: list[str], results_e2: list[str]
) -> tuple[
    list[list[dict]],
    list[list[dict]],
    dict[str, list[dict]],
    dict[str, list[dict]],
]:
    """Load all experiment results for both E1 and E2."""
    logger.info("Loading all results...")

    loaded_results_e1 = [load_results(r) for r in results_e1]
    loaded_results_e2 = [load_results(r) for r in results_e2]

    loaded_by_model_e1 = dict(zip(datasets, loaded_results_e1))

    loaded_by_model_e2 = {
        ds: res
        for ds, res in zip(datasets, loaded_results_e2)
        if ds not in ["oracle", "oasst2"]
    }

    return loaded_results_e1, loaded_results_e2, loaded_by_model_e1, loaded_by_model_e2


def compute_all_m_scores(
    loaded_results_e1: list[list[dict]],
    loaded_results_e2: list[list[dict]],
    datasets: list[str],
    legacy: bool = False,
    weights: Optional[tuple[int, ...]] = None,
) -> tuple[list[MScoreResult], list[MScoreResult]]:
    """Compute M-scores for all datasets in both experiments."""
    logger.info("Computing M-scores for all datasets...")

    m_scores_e1 = [
        calculate_m_score(res, ds, legacy=legacy, weights=weights)
        for res, ds in zip(loaded_results_e1, datasets)
    ]

    m_scores_e2 = [
        calculate_m_score(res, ds, legacy=legacy, weights=weights)
        for res, ds in zip(loaded_results_e2, datasets)
    ]

    return m_scores_e1, m_scores_e2


def compute_mg_counts(
    loaded_results_e1: list[list[dict]],
    loaded_results_e2: list[list[dict]],
    filter_epicenes: bool = False,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Compute MG noun counts for all datasets."""
    logger.info("Computing MG counts...")

    mg_counts_e1 = [
        get_mg_count(res, filter_epicenes=filter_epicenes) for res in loaded_results_e1
    ]
    mg_counts_e2 = [
        get_mg_count(res, filter_epicenes=filter_epicenes)
        for res, ds in zip(
            loaded_results_e2, [r[0]["dataset"] for r in loaded_results_e2]
        )
        if ds not in ["oracle", "oasst2"]
    ]

    mg_total_e1 = get_mg_total(mg_counts_e1)
    mg_total_e2 = get_mg_total(mg_counts_e2)

    return mg_counts_e1, mg_counts_e2, mg_total_e1, mg_total_e2


# plot
def generate_all_plots(
    m_scores_e1: list[MScoreResult],
    m_scores_e2: list[MScoreResult],
    loaded_results_e1: list[list[dict]],
    loaded_results_e2: list[list[dict]],
    mg_counts_e1: list[pd.DataFrame],
    mg_counts_e2: list[pd.DataFrame],
    mg_total_e1: pd.DataFrame,
    mg_total_e2: pd.DataFrame,
    datasets: list[str],
    is_e2: bool = False,
    mg_count_range: list[int] = [0, 50],
    z_score: float = 1.7,
    pdf: bool = False,
    legacy: bool = False,
    filter_epicenes: bool = False,
):
    """Generate all plots."""
    logger.info("Generating plots...")

    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # convert MScoreResults to dicts for plotting
    m_scores_e1_dicts = mscores_to_dicts(m_scores_e1)
    # m_scores_e2_dicts = mscores_to_dicts(m_scores_e2)

    # e2_m_score_map = {
    #     m.dataset: mscore_to_dict(m)
    #     for m in m_scores_e2
    #     if m.dataset not in ["oracle", "oasst2"]
    # }

    # visualize_m_scores(
    #     m_scores_e1_dicts,
    #     output_file=str(plot_dir / "m_scores.svg"),
    #     m_score_results_e2_map=e2_m_score_map,
    #     pdf=pdf,
    #     legacy=legacy,
    # )

    if is_e2:
        datasets_filtered = [d for d in datasets if d not in ["oracle", "oasst2"]]
        loaded_filtered = [
            res
            for res, ds in zip(loaded_results_e2, datasets)
            if ds not in ["oracle", "oasst2"]
        ]
        visualize_classes(
            loaded_filtered,
            datasets_filtered,
            e2=True,
            output_file=str(plot_dir / "classes_e2.svg"),
            pdf=pdf,
        )
    else:
        visualize_classes(
            loaded_results_e1,
            datasets,
            e2=False,
            output_file=str(plot_dir / "classes_e1.svg"),
            pdf=pdf,
        )

    e2_data_map = {
        ds: (m, res)
        for ds, m, res in zip(datasets, m_scores_e2, loaded_results_e2)
        if ds not in ["oracle", "oasst2"]
    }

    e2_data_map = {}
    for ds, m_score, loaded_res in zip(datasets, m_scores_e2, loaded_results_e2):
        if ds not in ["oracle", "oasst2"]:
            e2_data_map[ds] = (mscore_to_dict(m_score), loaded_res)

    visualize_bias_rate(
        m_scores_e1_dicts,
        loaded_results_e1,
        datasets,
        e2_data_map=e2_data_map,
        output_file=str(plot_dir / "bias_rate.svg"),
        pdf=pdf,
    )

    loaded_no_human_e1 = [
        res
        for res, ds in zip(loaded_results_e1, datasets)
        if ds not in ["oracle", "oasst2"]
    ]

    loaded_no_human_e2 = [
        res
        for res, ds in zip(loaded_results_e2, datasets)
        if ds not in ["oracle", "oasst2"]
    ]

    visualize_marker_types(
        loaded_no_human_e1,
        loaded_no_human_e2,
        output_file=str(plot_dir / "language_markers.svg"),
        pdf=pdf,
    )

    # loaded_no_human_e1 = [
    #     res
    #     for res, ds in zip(loaded_results_e1, datasets)
    #     if ds not in ["oracle", "oasst2"]
    # ]

    # visualize_marker_types(
    #     loaded_no_human_e1,
    #     output_file=str(plot_dir / "language_markers_e1.svg"),
    #     pdf=pdf,
    # )

    # if is_e2:
    #     loaded_no_human_e2 = [
    #         res
    #         for res, ds in zip(loaded_results_e2, datasets)
    #         if ds not in ["oracle", "oasst2"]
    #     ]
    #     visualize_marker_types(
    #         loaded_no_human_e2,
    #         e2=True,
    #         output_file=str(plot_dir / "language_markers_e2.svg"),
    #         pdf=pdf,
    #     )

    mg_counts = mg_counts_e2 if is_e2 else mg_counts_e1
    mg_total = mg_total_e2 if is_e2 else mg_total_e1
    suffix = "e2" if is_e2 else "e1"

    visualize_mg_count(
        mg_counts,
        total_df=mg_total,
        e2=is_e2,
        model_specific=True,
        rangee=mg_count_range,
        z_score=z_score,
        output_file=str(plot_dir / f"masc_gen_nouns_{suffix}.svg"),
        pdf=pdf,
        filter_epicenes=filter_epicenes,
    )

    logger.info("All plots generated")


def validate_args_order(
    datasets: list[str], results: list[str], results_mgonly: list[str]
):
    """Validate that datasets and results lists are properly aligned."""
    if not (len(datasets) == len(results) == len(results_mgonly)):
        sys.exit(
            "ERROR: The number of --datasets, --results, and --results_mgonly must match."
        )

    for i, dataset in enumerate(datasets):
        if dataset not in results[i] or dataset not in results_mgonly[i]:
            sys.exit(
                f"ERROR: Order mismatch at position {i}.\n"
                f"   Dataset: {dataset}\n"
                f"   Result: {results[i]}\n"
                f"   Result (mgonly): {results_mgonly[i]}"
            )


def setup_default_paths() -> tuple[list[str], list[str], list[str]]:
    """Set up default paths for datasets and results."""
    datasets = [name for _, name in MODEL_TYPE_NAMES]
    results = []
    results_mgonly = []

    for model_type, model_name in MODEL_TYPE_NAMES:
        if model_name == "oasst2":
            path = "instr_outputs_mg_results/real/human/oasst2_assistant_results_final.json"
            results.append(path)
            results_mgonly.append(path)
        elif model_name == "oracle":
            path = (
                "instr_outputs_mg_results/real/human/oracle_output_results_final.json"
            )
            results.append(path)
            results_mgonly.append(path)
        else:
            results.append(
                f"instr_outputs_mg_results/real/{model_type}/{model_name}_response_results_final.json"
            )
            results_mgonly.append(
                f"instr_outputs_mg_results/real/{model_type}/{model_name}_response_results_mgonly_final.json"
            )

    return datasets, results, results_mgonly


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze gender bias in language model outputs"
    )

    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        help="Paths to JSON files containing results from Experiment 1.",
    )

    parser.add_argument(
        "--results_mgonly",
        type=str,
        nargs="+",
        help="Paths to JSON files containing results from Experiment 2 (MG only).",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Names of datasets ordered by the order of result files.",
    )

    parser.add_argument(
        "--e2",
        action="store_true",
        help="Show experiment 2 plots. Applies to plots: Classes, Z-Plot MG Count. Other plots show both experiments' results simultaneously",
    )

    parser.add_argument(
        "--mg_count_range",
        type=int,
        nargs=2,
        default=[0, 50],
        help="(Z-Plot MG Count only) Range of MG counts to consider.",
    )

    parser.add_argument(
        "--z_score",
        type=float,
        default=1.7,
        help="(Z-Plot MG Count only) Z-Score threshold for outlier detection.",
    )

    parser.add_argument(
        "--filter-epicenes",
        action="store_true",
        help="(Z-Plot MG Count only) Filter out epicene nouns from MG count analysis.",
    )

    parser.add_argument(
        "--weights",
        nargs=5,
        type=float,
        help="(shelved) Custom weights for [mg, inclusive, inclusive_greetings, neutral, neutral_pronouns] in MScore formula.",
    )

    parser.add_argument(
        "--ci",
        type=str,
        choices=["br", "pareto", "markers"],
        help="Compute and display confidence intervals for selected metrics.",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also generate PDF versions of the plots (requires svg2pdf installed).",
    )

    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy MScore formula (MG/HN ratio only).",
    )

    parser.add_argument(
        "--n_boot_pareto",
        type=int,
        default=10000,
        help="Number of bootstrap iterations for Pareto stability analysis.",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    parser.add_argument(
        "--pareto_only",
        action="store_true",
        help="Run Pareto analysis only and exit.",
    )

    return parser.parse_args()


def setup_logging(log_level: str):
    """Configure logging with colored output."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    coloredlogs.install(level=log_level, logger=logger)


def main():
    """Main execution pipeline."""
    args = parse_arguments()
    setup_logging(args.log_level)

    # setup paths
    if not all([args.results, args.results_mgonly, args.datasets]):
        logger.info("No arguments provided, using default paths")
        datasets, results_e1, results_e2 = setup_default_paths()
    else:
        datasets = args.datasets
        results_e1 = args.results
        results_e2 = args.results_mgonly

    all_datasets = [name for _, name in MODEL_TYPE_NAMES]
    if any(ds not in all_datasets for ds in datasets):
        raise ValueError(f"Datasets must be one of {all_datasets}. Got: {datasets}")

    validate_args_order(datasets, results_e1, results_e2)

    if args.legacy:
        logger.info("Using LEGACY MScore formula")

    weights = tuple(args.weights) if args.weights else None
    if weights:
        logger.info(f"Using custom weights: {weights}")

    # load all data
    (
        loaded_results_e1,
        loaded_results_e2,
        loaded_by_model_e1,
        loaded_by_model_e2,
    ) = load_all_results(datasets, results_e1, results_e2)

    if args.ci == "br":
        # generate bias rate CSV with CIs and exit
        generate_bias_rate_ci_csv(
            loaded_results_e1,
            loaded_results_e2,
            datasets,
            output_file="analyses/bias_rates.csv",
            n_boot=10000,
        )
        sys.exit(0)

    elif args.ci == "markers":
        generate_marker_ci_csv(
            loaded_results_e1,
            loaded_results_e2,
            datasets,
            output_file="analyses/marker_rates.csv",
            n_boot=10000,
        )
        sys.exit(0)

    # compute M-scores
    m_scores_e1, m_scores_e2 = compute_all_m_scores(
        loaded_results_e1, loaded_results_e2, datasets, args.legacy, weights
    )

    # compute MG counts
    mg_counts_e1, mg_counts_e2, mg_total_e1, mg_total_e2 = compute_mg_counts(
        loaded_results_e1, loaded_results_e2, filter_epicenes=args.filter_epicenes
    )

    # if Pareto only, run before other plots and exit
    # run_pareto_analysis() contains function for its own plot
    if args.pareto_only:
        run_pareto_analysis(
            m_scores_e1,
            m_scores_e2,
            loaded_by_model_e1,
            loaded_by_model_e2,
            ci=args.ci == "pareto",
            n_boot=args.n_boot_pareto,
            pdf=args.pdf,
        )
        logger.info("Pareto analysis complete. Exiting.")
        return

    # generate all plots
    generate_all_plots(
        m_scores_e1,
        m_scores_e2,
        loaded_results_e1,
        loaded_results_e2,
        mg_counts_e1,
        mg_counts_e2,
        mg_total_e1,
        mg_total_e2,
        datasets,
        is_e2=args.e2,
        mg_count_range=args.mg_count_range,
        z_score=args.z_score,
        pdf=args.pdf,
        legacy=args.legacy,
        filter_epicenes=args.filter_epicenes,  # only affects MG count plot
    )

    # if pareto has not been ran yet, run it now
    run_pareto_analysis(
        m_scores_e1,
        m_scores_e2,
        loaded_by_model_e1,
        loaded_by_model_e2,
        ci=args.ci == "pareto",
        n_boot=args.n_boot_pareto,
        pdf=args.pdf,
    )


if __name__ == "__main__":
    main()
