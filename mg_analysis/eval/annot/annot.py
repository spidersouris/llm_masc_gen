import json
import csv
import re
import ast
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def load_json_data(json_file: str) -> dict:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def prepare_annotation_tsv(data: dict, output_tsv: str) -> None:
    dataset_counters = {
        "hh_rlhf_filtered_df": 0,
        "alpaca_instruct_df": 0,
    }

    max_entries_per_dataset = 250

    with open(output_tsv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            [
                "dataset",
                "text_index",
                "text_index_dataset",
                "text",
                "original_annotation",
                "annotator_annotation",
            ]
        )

        for entry in data:
            try:
                text_index_dataset = entry["text_index_dataset"]
                dataset_name = re.sub(r"_\d+$", "", text_index_dataset)

                if (
                    dataset_name in dataset_counters
                    and dataset_counters[dataset_name] >= max_entries_per_dataset
                ):
                    continue

                if dataset_name in dataset_counters:
                    dataset_counters[dataset_name] += 1

                dataset = entry["dataset"]
                text_index = entry["text_index"]
                text = entry["text"]
                original_annotation = entry["real_human_nouns"]

                writer.writerow(
                    [
                        dataset,
                        text_index,
                        text_index_dataset,
                        text,
                        original_annotation,
                        "",
                    ]
                )
            except KeyError as e:
                dataset_counters[dataset_name] -= 1
                print(f"KeyError: {e}, {entry}")
                continue


def load_annotations(tsv_file: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_file, sep="\t")

    # check that every dict is formatted correctly
    for idx, row in df.iterrows():
        original_annotation = row["original_annotation"]
        annotator_annotation = row["annotator_annotation"]

        try:
            og_annot_dict = ast.literal_eval(original_annotation)
        except (SyntaxError, ValueError):
            raise ValueError(
                f"Invalid JSON format in 'original_annotation' column at index {idx}."
            )

        try:
            ant_annot_dict = ast.literal_eval(annotator_annotation)
        except (SyntaxError, ValueError):
            raise ValueError(
                f"Invalid JSON format in 'annotator_annotation' column at index {idx}."
            )

        if any(v not in [0, 1] for v in og_annot_dict.values()):
            raise ValueError(
                f"Invalid values in 'original_annotation' column at index {idx}."
            )

        if any(v not in [0, 1] for v in ant_annot_dict.values()):
            raise ValueError(
                f"Invalid values in 'annotator_annotation' column at index {idx}."
            )

    return df


def calculate_kappa(
    annotations1: pd.DataFrame,
    annotations2: pd.DataFrame | None,
    compare_gpt: bool = False,
) -> float:
    ann1 = annotations1["annotator_annotation"]
    ann2 = (
        annotations2["annotator_annotation"]
        if not compare_gpt
        else annotations1["original_annotation"]
    )

    differences = {
        key: (ann1[key], ann2[key])
        for key in list(ann1.keys())
        if ann1[key] != ann2[key]
    }

    print(f"Found {len(differences)} differences in annotations:")
    for idx, (value_1, value_2) in differences.items():
        print(f"{idx+2}:\nAnnotator 1 = {value_1}\nAnnotator 2 = {value_2}")

    return cohen_kappa_score(ann1, ann2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to handle annotations.")
    parser.add_argument(
        "--create-tsv",
        nargs=2,
        metavar=("json_file", "output_tsv"),
        help="Create TSV file for annotation.",
    )
    parser.add_argument(
        "--kappa",
        nargs=2,
        metavar=("annotator1_tsv", "annotator2_tsv"),
        help="Calculate Kappa score between two annotators.",
    )
    parser.add_argument(
        "--kappa_gpt",
        nargs=1,
        metavar=("agreed_tsv"),
        help="Calculate Kappa score between GPT-4o mini and agreed annotation.",
    )
    args = parser.parse_args()

    if args.create_tsv:
        json_data = load_json_data(args.create_tsv[0])
        prepare_annotation_tsv(json_data, args.create_tsv[1])
        exit()

    if args.kappa:
        annotations1 = load_annotations(args.kappa[0])
        annotations2 = load_annotations(args.kappa[1])
        calculate_kappa(annotations1, annotations2)

    if args.kappa_gpt:
        annotations1 = load_annotations(args.kappa_gpt[0])
        calculate_kappa(annotations1, None, compare_gpt=True)
        exit()
