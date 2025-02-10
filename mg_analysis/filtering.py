import pandas as pd
import time
import spacy
import numpy as np
from tqdm import tqdm
from spacy.tokens import Doc
import argparse

from load_lists import get_masc_gen_list, get_neutrals_list

nlp_main = spacy.load("fr_dep_news_trf")  # main processing
nlp_ent = spacy.load("fr_core_news_lg")  # NER


masc_gen_list = get_masc_gen_list()
neutrals_dict = get_neutrals_list()


def _get_first_names(file_path: str = "first_names.csv", col_name: str = "Prenoms"):
    return pd.read_csv(file_path, delimiter=";")[col_name].tolist()


def has_interr_pron(doc: Doc):
    if not isinstance(doc, Doc):
        raise TypeError("doc must be a spacy Doc object")

    for token in doc:
        if token.text.lower() == "qui" and token.pos_ == "PRON":
            if token.dep_ in {"nsubj", "obj", "ROOT"}:
                if "Int" in token.morph.get("PronType", None):
                    return True

    return False


def has_spec_det(doc: Doc, masc_gen_words: list[str]):
    spec_dets = ["mon", "ton", "son", "sa", "notre", "votre", "leur", "ce", "cet"]
    if not isinstance(doc, Doc):
        raise TypeError("doc must be a spacy Doc object")

    for token in doc:
        # print(token.text, token.dep_, token.pos_, token.lemma_)
        if token.text.lower() in spec_dets and token.pos_ == "DET":
            if token.nbor():
                if (
                    token.nbor().lemma_ in masc_gen_words
                    and token.nbor().pos_ == "NOUN"
                ):
                    return True
                elif token.nbor().pos_ == "ADJ":
                    if token.i + 2 < len(token.doc):
                        if token.nbor(2):
                            if (
                                token.nbor(2).lemma_ in masc_gen_words
                                and token.nbor(2).pos_ == "NOUN"
                            ):
                                return True

    return False


def has_person_ent(doc: Doc):
    if not isinstance(doc, Doc):
        raise TypeError("doc must be a spacy Doc object")

    first_names = _get_first_names()

    for ent in doc.ents:
        if ent.label_ == "PER":
            return True
        elif ent.label_ == "MISC" and any(
            x for x in str(ent).split() if x.strip() in first_names
        ):
            return True

    return False


def has_masc_gen(doc: Doc, masc_gen_words: list[str]):
    if not isinstance(doc, Doc):
        raise TypeError("doc must be a spacy Doc object")

    for token in doc:
        if token.lemma_ in masc_gen_words and token.pos_ == "NOUN":
            return True

    return False


def has_neutral(doc: Doc, neutrals: list[str] = neutrals_dict):
    if not isinstance(doc, Doc):
        raise TypeError("doc must be a spacy Doc object")

    for token in doc:
        if token.lemma_ in neutrals and token.pos_ == "NOUN":
            return True

    return False


def save_to_pickle(df, dataset):
    df.to_pickle(f"dfs/{dataset}/{dataset}_filtered_df.pkl")
    print(f"\n\nSuccessfully saved to dfs/{dataset}/{dataset}_filtered_df.pkl")


def filter_human_in_out(
    df: pd.DataFrame,
    nlp_main: spacy.language.Language = nlp_main,
    nlp_ent: spacy.language.Language = nlp_ent,
    masc_gen_words: list[str] = masc_gen_list,
    dataset: str | None = None,
    batch_size: int = 32,
    return_df: bool = False,
    filter_masc_gen: bool = False,
    neutral_col: bool = False,
    neutral_words: list[str] = neutrals_dict,
    check_interr_det: bool = True,
):
    modified_df = df.copy()

    if dataset == "oracle":
        columns_to_process = [
            col
            for col in modified_df.columns
            if col.lower() in ["instruction", "output"]
        ]
    elif dataset in ["oasst2", "hh_rlhf", "alpaca"]:
        columns_to_process = [
            col for col in modified_df.columns if "content" in col.lower()
        ]
    elif dataset in [
        "gemini",
        "gpt4o_mini",
        "claude-3-haiku",
        "ministral",
        "llama",
        "mistral-small",
    ]:
        columns_to_process = ["response"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if neutral_col and neutral_words is None:
        raise ValueError("neutral_words must be provided if neutral_col is True")

    notna_sum = modified_df.notna().sum()
    print(
        "Number of non-NA values in each column BEFORE filter_human_in_out ",
        "with dataset %s:\n%s",
        dataset,
        notna_sum,
    )

    start_time = time.time()

    for col in columns_to_process:
        tqdm_desc = f"Processing column '{col}' | Dataset: {dataset}"
        progress_bar = tqdm(total=len(modified_df), desc=tqdm_desc, leave=True)

        non_na_indices = modified_df[col].index
        non_na_texts = modified_df[col].tolist()

        for batch_start in range(0, len(non_na_texts), batch_size):
            batch_end = batch_start + batch_size
            batch_indices = non_na_indices[batch_start:batch_end]
            batch_texts = non_na_texts[batch_start:batch_end]

            docs_main = list(nlp_main.pipe(batch_texts, batch_size=batch_size))

            for idx, doc_main in zip(batch_indices, docs_main):
                cell_content = str(modified_df.at[idx, col])  # Original content

                # check for interrogative pronouns or specific determiners
                # if check_interr_det and (has_interr_pron(doc_main)
                # or has_spec_det(doc_main, masc_gen_words)):
                if check_interr_det and has_spec_det(doc_main, masc_gen_words):
                    # print("Found interr or det, skipping")
                    print("Found interr or det, skipping")
                    modified_df.at[idx, col] = np.nan
                    progress_bar.update(1)
                    continue

                if filter_masc_gen and has_masc_gen(doc_main, masc_gen_words):
                    print("Found masc gen, skipping")
                    modified_df.at[idx, col] = np.nan
                    progress_bar.update(1)
                    continue

                # check for person entities
                if nlp_ent is not None:
                    doc_ent = nlp_ent(cell_content)  # Process with entity model
                    if has_person_ent(doc_ent):
                        # print("Found person, skipping: %s",
                        # [ent.text for ent in doc_ent.ents])
                        print(
                            "Found person, skipping: %s",
                            [ent.text for ent in doc_ent.ents],
                        )
                        modified_df.at[idx, col] = np.nan
                        progress_bar.update(1)
                        continue

                # neutral check
                if neutral_col:
                    if has_neutral(doc_main, neutrals_dict):
                        modified_df.at[idx, "neutral"] = 1
                    else:
                        modified_df.at[idx, "neutral"] = 0

                # Update progress bar after processing each cell
                progress_bar.update(1)

        # Close the progress bar for this column
        progress_bar.close()

    if "neutral" in modified_df.columns:
        modified_df["neutral"] = modified_df["neutral"].astype(int, errors="ignore")

    end_time = time.time()
    elapsed_time_min = (end_time - start_time) / 60
    print(f"\n\nTotal time taken: {elapsed_time_min:.2f} minutes")

    notna_sum = modified_df.notna().sum()
    print(
        "Number of non-NA values in each column AFTER filter_human_in_out ",
        "with dataset %s:\n%s",
        dataset,
        notna_sum,
    )

    save_to_pickle(modified_df, dataset)

    if return_df:
        return modified_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter instruction datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to filter",
        choices=["oasst2", "oracle", "hh_rlhf", "alpaca"],
    )

    args = parser.parse_args()

    if args.dataset == "oasst2":
        oasst2_df = pd.read_pickle("dfs/oasst2/oasst2_df.pkl")
        filter_human_in_out(df=oasst2_df, dataset="oasst2", return_df=False)
    elif args.dataset == "oracle":
        oracle_df = pd.read_pickle("dfs/oracle/oracle_df.pkl")
        filter_human_in_out(df=oracle_df, dataset="oracle", return_df=False)
    elif args.dataset == "hh_rlhf":
        hh_rlhf_df = pd.read_pickle("dfs/hh_rlhf/hh_rlhf_df.pkl")
        filter_human_in_out(df=hh_rlhf_df, dataset="hh_rlhf", return_df=False)
    elif args.dataset == "alpaca":
        alpaca_df = pd.read_pickle("dfs/alpaca/alpaca_df.pkl")
        filter_human_in_out(df=alpaca_df, dataset="alpaca", return_df=False)
