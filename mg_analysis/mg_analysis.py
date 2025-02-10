import json
import os
import re
import argparse
import pandas as pd
import html as ihtml
from bs4 import BeautifulSoup
from typing import List, Dict
from spacy.tokens import Doc
from tqdm import tqdm
from load_lists import (
    get_masc_gen_list,
    get_epicene_list,
    get_human_list,
)
from filtering import nlp_main

epicene_list = get_epicene_list()
masc_gen_list = get_masc_gen_list()
human_list = get_human_list()

incl_greetings = [
    "tous et toutes",
    "toutes et tous",
    "messieurs, dames",
    "mesdames et messieurs",
    "messieurs dames",
    "messieurs et mesdames",
]
incl_pairs = [
    "ceux et celles",
    "celles et ceux",
    "ceux ou celles",
    "celles ou ceux",
    "celui ou celle",
    "celle ou celui",
    "ils et elles",
    "elles et ils",
    "eux et elles",
    "elles et eux",
    "ils ou elles",
    "elles ou ils",
    "eux ou elles",
    "elles ou eux",
    "elle ou il",
    "il ou elle",
    "la ou le",
    "un ou une",
    "le ou la",
    "une ou un",
]
neutral_prons = ["iel", "iels", "ielle", "ielles", "celleux", "elleux", "ille", "illes"]
uppers = sorted(
    [
        "E",
        "ES",
        "SE",
        "SES",
        "EUSE",
        "EUSES",
        "EURE",
        "EURES",
        "RICE",
        "RICES",
        "ICE",
        "ICES",
        "NE",
        "NES",
        "ÈRE",
        "ÈRES",
        "ERE",
        "ERES",
        "TE",
        "TES",
        "VE",
        "VES",
        "LLE",
        "LLES",
        "LE",
        "LES",
    ],
    key=len,
    reverse=True,
)  # match longest first
seps = ["·", "‧", ".", "⋅", "•", "∙", "/", "("]

# do not match programming keywords with parentheses
code_words = [
    "print",
    "print(e)",
    "print(e",
    "del",
    "del(e)",
    "del(e",
    "except",
    "except(e",
    "except(e)",
    "match(e)",
    "match",
    "match(e",
    "return",
    "return(e",
    "return(e)",
]


def load_df(df_path):
    if not df_path.endswith(".pkl"):
        raise ValueError("df_path must be a .pkl file")

    df = pd.read_pickle(df_path)
    return df


def remove_html(text):
    text = BeautifulSoup(ihtml.unescape(text)).text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def count_terms(
    doc: Doc,
    incl_greetings: List[str],
    incl_pairs: List[str],
    neutral_prons: List[str],
    uppers: List[str],
    separators: List[str],
    human_list: List[str],
    masc_gen_list: List[str],
    epicene_list: List[str],
) -> Dict:
    """
    Analyze a spacy Doc object to count inclusive terms, neutral terms,
    masculine generics, and human nouns.
    """
    incl_greetings_pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, incl_greetings)) + r")\b", re.IGNORECASE
    )
    separator_pattern = re.compile(
        r"\b[A-zÀ-ú]+(("
        + "|".join(map(re.escape, separators))
        + r")("
        + "|".join(map(re.escape, uppers))
        + r")(\)\[A-zÀ-ú]*)?)\b",
        re.IGNORECASE,
    )
    upper_pattern = re.compile(
        r"\b[A-zÀ-ú]+?(" + "|".join(map(re.escape, uppers)) + r")\b"
    )
    incl_pairs_pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, incl_pairs)) + r")\b", re.IGNORECASE
    )
    neutral_prons_pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, neutral_prons)) + r")\b", re.IGNORECASE
    )

    masc_gen_dets = ["ce", "cet", "un", "le", "du"]
    neutrals_dict = ["personne", "personnes", "individu", "individus", "gens"]

    incl_greetings_logs = []
    incl_pairs_logs = []
    neutral_prons_logs = []
    separator_logs = []
    upper_logs = []
    neutral_logs = []
    masc_gen_logs = []

    human_nouns = []

    for sent in doc.sents:
        matches = incl_greetings_pattern.findall(sent.text)
        if matches:
            for m in matches:
                incl_greetings_logs.append({"sentence": sent.text, "incl_greeting": m})

        matches = incl_pairs_pattern.findall(sent.text)
        if matches:
            for m in matches:
                incl_pairs_logs.append({"sentence": sent.text, "incl_pair": m})

        matches = neutral_prons_pattern.findall(sent.text)
        if matches:
            for m in matches:
                neutral_prons_logs.append({"sentence": sent.text, "neutral_pron": m})

    for token in doc:
        # Count human-related nouns
        if token.pos_ == "NOUN" and (
            token.lemma_ in human_list
            or (separator_pattern.match(token.text) and token.text not in code_words)
            or (upper_pattern.match(token.text) and token.text != token.text.upper())
        ):
            human_nouns.append(token.text)

            sep_match = separator_pattern.match(token.text)
            if sep_match:
                separator_logs.append(
                    {"token": token.text, "separator": sep_match.group(1)}
                )

            upper_match = upper_pattern.match(token.text)
            if upper_match and token.text != token.text.upper():
                upper_logs.append({"token": token.text, "upper": upper_match.group(1)})

            for neutral in neutrals_dict:
                if token.lemma_ == neutral:
                    neutral_logs.append({"token": token.text, "neutral": neutral})

            for masc_gen in masc_gen_list:
                if token.lemma_ == masc_gen:
                    masc_gen_logs.append(
                        {"token": token.text, "masc_gen": masc_gen, "from_epicene": 0}
                    )

            for epicene in epicene_list:
                if token.lemma_ == epicene:
                    if token.i - 1 >= 0:
                        if token.nbor(-1):
                            if (
                                token.nbor(-1).pos_ == "DET"
                                and token.nbor(-1).text in masc_gen_dets
                            ):
                                print(
                                    f"Found epicene word {epicene} with masc_gen_det ",
                                    f"{token.nbor(-1).text}",
                                )
                                masc_gen_logs.append(
                                    {
                                        "token": token.text,
                                        "masc_gen": epicene,
                                        "from_epicene": 1,
                                    }
                                )

    return {
        "incl_greetings_logs": incl_greetings_logs,
        "incl_pairs_logs": incl_pairs_logs,
        "neutral_prons_logs": neutral_prons_logs,
        "separator_logs": separator_logs,
        "upper_logs": upper_logs,
        "neutral_logs": neutral_logs,
        "masc_gen_logs": masc_gen_logs,
        "human_nouns": human_nouns,
        "human_noun_count": len(human_nouns),
    }


def analyze_dataframe(
    df,
    content_type: str,
    dataset: str,
    is_real: bool,
    incl_greetings: list[str] = incl_greetings,
    incl_pairs: list[str] = incl_pairs,
    neutral_prons: list[str] = neutral_prons,
    uppers: list[str] = uppers,
    separators: list[str] = seps,
    human_list: list[str] = human_list,
    masc_gen_list: list[str] = masc_gen_list,
    epicene_list: list[str] = epicene_list,
    nlp=nlp_main,
    batch_size: int = 32,
    use_tqdm: bool = True,
    save_to_file: bool = True,
    id_col: bool = False,
) -> None:
    """
    Analyze a dataframe to count inclusive, neutral,
    and masculine generic terms in the instructions/outputs.
    """
    results = []
    texts_to_process = []
    text_indices = []
    current_text_idx = 0

    dataset_groups = {
        "human_instr": ["oracle", "oasst2"],
        "llm_prop": ["gemini", "gpt4o_mini", "claude-3-haiku"],
        "llm_local": ["ministral", "llama", "mistral-small"],
    }

    dataset_group = None
    for group, datasets in dataset_groups.items():
        if dataset in datasets:
            dataset_group = group
            break

    if dataset_group is None:
        raise ValueError(f"dataset {dataset} missing from dataset_groups")

    for idx, row in df.iterrows():
        for col in row.index:
            if content_type in col.lower():
                text = row[col]
                if not pd.isna(text) and text not in texts_to_process:
                    # remove possible HTML
                    if any(sep in text for sep in separators):
                        texts_to_process.append(remove_html(text))
                    else:
                        texts_to_process.append(text)
                    if id_col and "id" in df.columns:
                        text_indices.append(row["id"])
                    else:
                        text_indices.append(idx)

    if use_tqdm:
        progress_bar = tqdm(total=len(texts_to_process), desc="Processing texts")
    else:
        progress_bar = None

    for i in range(0, len(texts_to_process), batch_size):
        batch_texts = texts_to_process[i : i + batch_size]
        batch_indices = text_indices[i : i + batch_size]
        docs = list(nlp.pipe(batch_texts))

        for doc, text, idx in zip(docs, batch_texts, batch_indices):
            analysis_result = count_terms(
                doc=doc,
                incl_greetings=incl_greetings,
                incl_pairs=incl_pairs,
                neutral_prons=neutral_prons,
                uppers=uppers,
                separators=separators,
                human_list=human_list,
                masc_gen_list=masc_gen_list,
                epicene_list=epicene_list,
            )
            results.append(
                {
                    "dataset": dataset,
                    "text_index": current_text_idx,
                    "text_index_dataset": (
                        dataset + "_" + str(current_text_idx) if not id_col else idx
                    ),
                    "text": text,
                    "incl_greetings_logs": analysis_result["incl_greetings_logs"],
                    "incl_pairs_logs": analysis_result["incl_pairs_logs"],
                    "neutral_prons_logs": analysis_result["neutral_prons_logs"],
                    "separator_logs": analysis_result["separator_logs"],
                    "upper_logs": analysis_result["upper_logs"],
                    "neutral_logs": analysis_result["neutral_logs"],
                    "masc_gen_logs": (
                        analysis_result["real_masc_gen_logs"]
                        if is_real
                        else analysis_result["masc_gen_logs"]
                    ),
                    "human_nouns": (
                        analysis_result["real_human_nouns"]
                        if is_real
                        else analysis_result["human_nouns"]
                    ),
                    "human_noun_count": (
                        analysis_result["real_human_noun_count"]
                        if is_real
                        else analysis_result["human_noun_count"]
                    ),
                }
            )

            current_text_idx += 1

        if progress_bar:
            progress_bar.update(len(batch_texts))

    if progress_bar:
        progress_bar.close()

    if save_to_file:
        real_folder = "real" if is_real else "unreal"

        os.makedirs(
            f"instr_outputs_mg_results/{real_folder}/{dataset_group}",
            exist_ok=True,
        )
        with open(
            f"instr_outputs_mg_results/{real_folder}/{dataset_group}/{dataset}_{content_type}_results.json",
            "w",
            encoding="utf8",
        ) as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(
            f"\n\nSuccessfully saved instr_outputs_mg_results/{real_folder}/{dataset_group}/{dataset}_{content_type}_results.json"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "df_path",
        type=str,
        help="Path to the filtered dataframe file (X_filtered_df.pkl)",
    )
    parser.add_argument(
        "--content_type",
        type=str,
        default="instruction",
        help="Type of content to analyze (instruction, output)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="oasst2",
        help="Dataset to analyze (oasst2, oracle, etc.)",
    )
    parser.add_argument(
        "--is_real",
        action="store_true",
        help="Whether the dataset is real (has been validated by GPT) or not",
    )

    args = parser.parse_args()

    df = load_df(args.df_path)
    analyze_dataframe(df, args.content_type, args.dataset, args.is_real)
