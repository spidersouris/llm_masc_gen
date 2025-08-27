import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from plot import (
    visualize_bias_rate,
    visualize_classes,
    visualize_m_scores,
    visualize_marker_types,
    visualize_mg_count,
)


def validate_args_order(
    datasets: list[str], results: list[str], results_mgonly: list[str]
):
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


def load_results(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf8") as f:
        return json.load(f)


def merge_json_results(*file_paths: str) -> list[dict]:
    merged_results = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping...")
            continue

        if not file_path.lower().endswith(".json"):
            print(f"Warning: File {file_path} is not a JSON file. Skipping...")
            continue

        try:
            results = load_results(file_path)
            merged_results.extend(results)
            print(f"Loaded {len(results)} results from {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON from {file_path}")
        except Exception as e:
            print(f"Unexpected error loading {file_path}: {e}")

    return merged_results


def get_mg_count(results: list[dict]) -> pd.DataFrame:
    masc_gen_nouns = []

    for dataset_result in results:
        dataset_name = dataset_result["dataset"]

        masc_gen_logs = dataset_result.get("real_masc_gen_logs", [])
        masc_gen_nouns.extend([log["masc_gen"] for log in masc_gen_logs])

    masc_gen_df = pd.DataFrame(masc_gen_nouns, columns=["noun"])
    masc_gen_df = masc_gen_df.value_counts().reset_index()
    masc_gen_df["dataset"] = dataset_name

    return masc_gen_df


def get_mg_total(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    combined_df = pd.concat(dfs, ignore_index=True)
    ranked_nouns = combined_df.groupby("noun")["count"].sum().reset_index()
    ranked_nouns = ranked_nouns.sort_values("count", ascending=False).reset_index(
        drop=True
    )
    ranked_nouns["rank"] = ranked_nouns.index + 1

    # switch rank col to 1st position
    cols = ["rank"] + [col for col in ranked_nouns.columns if col != "rank"]
    ranked_nouns = ranked_nouns[cols]

    return ranked_nouns


def calculate_m_score(
    results: list[dict],
    dataset: str,
    content_type: str | None = None,
    is_real: bool = True,
):
    """
    Calculate the M score (ratio of masculine generic words to human words)

    Args:
        results (list): Results from the JSON analysis.

    Returns:
        dict: Dictionary containing overall M score and detailed statistics.
    """
    total_human_nouns = 0
    total_masc_gen_nouns = 0
    total_neut = 0
    total_incl = 0
    total_incl_greetings = 0
    total_neutral_prons = 0

    m_scores = []

    # human_nouns = set()
    # masc_gen_nouns = set()

    fp_idx_e1 = [
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
    fp_idx_e2 = [
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

    human_noun_count_label = "real_human_nouns_count" if is_real else "human_noun_count"
    masc_gen_logs_label = "real_masc_gen_logs" if is_real else "masc_gen_logs"

    for result in results:
        human_noun_count = result.get(human_noun_count_label, 0) or 0
        masc_gen_count = len(result.get(masc_gen_logs_label) or [])
        incl_greetings_count = len(result.get("incl_greetings_logs", []))
        incl_pairs_count = len(result.get("incl_pairs_logs", []))
        neutral_prons_count = len(result.get("neutral_prons_logs", []))
        separator_count = len(result.get("separator_logs", []))
        # it seems LLMs do not use this kind of gender-fair writing strategy
        # (and automatic detection leads to FPs), so let's skip it by default
        # upper_count = len(result.get("upper_logs", []))
        neutral_count = len(result.get("neutral_logs", []))

        # merge incl_pairs and separators as one
        # incl_greetings stays separate b/c not much impact
        inclusive_count = incl_pairs_count + separator_count

        # shouldn't happen
        if masc_gen_count > human_noun_count:
            masc_gen_count = human_noun_count

        total_human_nouns += human_noun_count
        total_masc_gen_nouns += masc_gen_count
        total_neut += neutral_count
        total_incl += inclusive_count
        total_incl_greetings += incl_greetings_count
        total_neutral_prons += neutral_prons_count

        incl_weight = 5
        incl_greetings_weight = 2
        neut_weight = 1
        neutral_prons_weight = 10

        m_score = (
            min(
                max(
                    (
                        (
                            masc_gen_count
                            - incl_weight * inclusive_count
                            - incl_greetings_weight * incl_greetings_count
                            - neut_weight * neutral_count
                            - neutral_prons_weight * neutral_prons_count
                        )
                        / human_noun_count
                    ),
                    -1,
                ),
                1,
            )
            if human_noun_count > 0
            else np.nan
        )
        # m_score = masc_gen_count / human_noun_count if human_noun_count > 0 else np.nan
        # m_score = len(loc_masc_gen_nouns) / len(loc_human_nouns) if len(loc_human_nouns) > 0 else np.nan

        is_text_biased = 1 if (masc_gen_count > 0 and human_noun_count > 0) else 0

        m_scores.append(
            {
                "text": result.get("text", ""),
                "is_text_biased": is_text_biased,
                "m_score": m_score,
                "human_noun_count": human_noun_count,
                "masc_gen_count": masc_gen_count,
                "neutral_count": neutral_count,
                "inclusive_count": inclusive_count,
            }
        )

    overall_m_score = (
        min(
            max(
                (
                    (
                        total_masc_gen_nouns
                        - incl_weight * total_incl
                        - incl_greetings_weight * total_incl_greetings
                        - neut_weight * total_neut
                        - neutral_prons_weight * total_neutral_prons
                    )
                    / total_human_nouns
                ),
                -1,
            ),
            1,
        )
        if total_human_nouns > 0
        else np.nan
    )
    # overall_m_score = total_masc_gen_nouns / total_human_nouns if total_human_nouns > 0 else np.nan
    # overall_m_score = len(masc_gen_nouns) / len(human_nouns) if len(human_nouns) > 0 else np.nan

    m_scores = [x for x in m_scores if not np.isnan(x["m_score"])]

    average_m_score = sum(entry["m_score"] for entry in m_scores) / len(m_scores)

    print(dataset, average_m_score)

    bias_rate = sum(entry["is_text_biased"] for entry in m_scores) / len(m_scores)

    m_score_results = {
        "dataset": dataset,
        "content_type": content_type,
        "overall_m_score": overall_m_score,
        "average_m_score": average_m_score,
        "bias_rate": bias_rate,
        "total_human_nouns": total_human_nouns,
        "total_masc_gen_nouns": total_masc_gen_nouns,
        "total_neut": total_neut,
        "total_incl": total_incl,
        "detailed_scores": m_scores,
    }

    return m_score_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        help="Names of the datasets ordered by the order of the results files.",
    )

    parser.add_argument(
        "--e2",
        action="store_true",
        help="Show experiment 2 plot. Applies to plots: Classes, Z-Plot MG Count. Other plots show both experiments' results simultaneously.",
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

    args = parser.parse_args()

    results = args.results
    results_mgonly = args.results_mgonly
    datasets = args.datasets
    is_e2 = args.e2
    mg_count_range = args.mg_count_range
    z_score = args.z_score

    all_datasets = [
        "oracle",
        "oasst2",
        "gemini",
        "gpt4o_mini",
        "claude-3-haiku",
        "llama",
        "ministral",
        "mistral-small",
    ]
    if any(ds not in all_datasets for ds in datasets):
        raise ValueError(f"Datasets must be one of {all_datasets}. Got: {datasets}")

    validate_args_order(datasets, results, results_mgonly)

    m_scores = []
    m_scores_e2 = []
    loaded_results = []
    loaded_results_e2 = []
    mg_counts = []
    mg_counts_e2 = []
    for res, ds in zip(results, datasets):
        loaded_res = load_results(res)
        loaded_results.append(loaded_res)
        m_score = calculate_m_score(loaded_res, ds)
        m_scores.append(m_score)
        mg_counts.append(get_mg_count(loaded_res))

    for res, ds in zip(results_mgonly, datasets):
        loaded_res = load_results(res)
        loaded_results_e2.append(loaded_res)
        m_score = calculate_m_score(loaded_res, ds)
        m_scores_e2.append(m_score)
        mg_counts_e2.append(get_mg_count(loaded_res))

    mg_total = get_mg_total(mg_counts)
    mg_total_e2 = get_mg_total(mg_counts_e2)

    if not os.path.exists("plots"):
        os.makedirs("plots")

    visualize_m_scores(
        m_scores,
        output_file="plots/m_scores.svg",
        # m_score_results_array_mgonly=results_mgonly,
        m_score_results_array_mgonly=m_scores_e2,
    )

    if is_e2:
        visualize_classes(
            loaded_results_e2, datasets, e2=is_e2, output_file="plots/classes.svg"
        )
    else:
        visualize_classes(
            loaded_results, datasets, e2=is_e2, output_file="plots/classes.svg"
        )

    num_empty_dicts = sum(dataset in datasets for dataset in ["oracle", "oasst2"])
    m_score_results_array_neut = [{}] * num_empty_dicts + [*m_scores_e2]
    general_results_neut = [{}] * num_empty_dicts + [*loaded_results_e2]

    visualize_bias_rate(
        m_scores,
        loaded_results,
        datasets,
        m_score_results_array_neut=m_score_results_array_neut,
        general_results_neut=general_results_neut,
        output_file="plots/bias_rate.svg",
    )
    visualize_marker_types(loaded_results, output_file="plots/language_markers.svg")

    if is_e2:
        visualize_mg_count(
            mg_counts_e2,
            total_df=mg_total_e2,
            e2=is_e2,
            model_specific=True,
            rangee=mg_count_range,
            z_score=z_score,
            output_file="plots/masc_gen_nouns.svg",
        )
    else:
        visualize_mg_count(
            mg_counts,
            total_df=mg_total,
            e2=is_e2,
            model_specific=True,
            rangee=mg_count_range,
            z_score=z_score,
            output_file="plots/masc_gen_nouns.svg",
        )
