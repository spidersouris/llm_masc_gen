import json
import os
import numpy as np
import argparse
from plot import (
    visualize_bias_rate,
    visualize_classes,
    visualize_m_scores,
    visualize_marker_types,
)


def load_results(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        return json.load(f)


def merge_json_results(*file_paths):
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


def calculate_m_score(
    results,
    dataset: str,
    content_type: str = "",
    is_real: bool = True,
):
    total_human_nouns = 0
    total_masc_gen_nouns = 0

    m_scores = []

    human_noun_count_label = "real_human_nouns_count" if is_real else "human_noun_count"
    masc_gen_logs_label = "real_masc_gen_logs" if is_real else "masc_gen_logs"

    for result in results:
        human_noun_count = result.get(human_noun_count_label, 0)
        masc_gen_count = len(result.get(masc_gen_logs_label, []))

        # shouldn't happen
        if masc_gen_count > human_noun_count:
            masc_gen_count = human_noun_count

        total_human_nouns += human_noun_count
        total_masc_gen_nouns += masc_gen_count

        m_score = masc_gen_count / human_noun_count if human_noun_count > 0 else np.nan

        is_text_biased = 1 if (masc_gen_count > 0 and human_noun_count > 0) else 0

        m_scores.append(
            {
                "text": result.get("text", ""),
                "is_text_biased": is_text_biased,
                "m_score": m_score,
                "human_noun_count": human_noun_count,
                "masc_gen_count": masc_gen_count,
            }
        )

    overall_m_score = (
        total_masc_gen_nouns / total_human_nouns if total_human_nouns > 0 else np.nan
    )

    m_scores = [x for x in m_scores if not np.isnan(x["m_score"])]

    average_m_score = sum(entry["m_score"] for entry in m_scores) / len(m_scores)

    bias_rate = sum(entry["is_text_biased"] for entry in m_scores) / len(m_scores)

    m_score_results = {
        "dataset": dataset,
        "content_type": content_type,
        "overall_m_score": overall_m_score,
        "average_m_score": average_m_score,
        "bias_rate": bias_rate,
        "total_human_nouns": total_human_nouns,
        "total_masc_gen_nouns": total_masc_gen_nouns,
        "detailed_scores": m_scores,
    }

    return m_score_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        help="Paths to JSON files containing results.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Names of the datasets ordered by the order of the results files.",
    )

    args = parser.parse_args()

    results = args.results
    datasets = args.datasets

    m_scores = []
    loaded_results = []
    for res, ds in zip(results, datasets):
        loaded_res = load_results(res)
        loaded_results.append(loaded_res)
        m_score = calculate_m_score(loaded_res, ds)
        m_scores.append(m_score)

    if not os.path.exists("plots"):
        os.makedirs("plots")

    visualize_m_scores(m_scores, output_file="plots/m_scores.svg")
    visualize_classes(loaded_results, datasets, output_file="plots/classes.svg")
    visualize_bias_rate(
        m_scores, loaded_results, datasets, output_file="plots/bias_rate.svg"
    )
    visualize_marker_types(loaded_results, output_file="plots/language_markers.svg")
