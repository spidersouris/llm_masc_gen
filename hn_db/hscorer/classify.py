from model_pipeline import ModelPipeline
from load_words import get_nouns_to_pipeline
from get_agreements import get_all_positive_df, get_disagreement_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import xgboost as xgb
import argparse
import sys


def load_models():
    with open("models/lr.pkl", "rb") as f:
        lr_model = joblib.load(f)

    xgb_model = xgb.Booster()
    xgb_model.load_model("models/xgb.json")

    tokenizer = AutoTokenizer.from_pretrained("spidersouris/hscore-balanced")
    tf_model = AutoModelForSequenceClassification.from_pretrained(
        "spidersouris/hscore-balanced"
    )

    return lr_model, xgb_model, tf_model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some words.")
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset",
        choices=["tlfi_recursive", "demonette"],
    )
    parser.add_argument(
        "sum_prob",
        type=int,
        help="Minimum sum of probabilities for agreement",
    )
    parser.add_argument("positive_df_path", type=str, help="Path to save positive df")
    parser.add_argument(
        "disagreement_df_path", type=str, help="Path to save disagreement df"
    )
    parser.add_argument(
        "--disagreement_type",
        type=str,
        default="two_pos_one_neg",
        help="Type of disagreement to check",
    )

    args = parser.parse_args()

    if args.dataset not in ["tlfi_recursive", "demonette"]:
        print("Invalid dataset. Please choose from 'tlfi_recursive' or 'demonette'")
        sys.exit(1)

    lr_model, xgb_model, tf_model, tokenizer = load_models()
    pipeline = ModelPipeline(lr_model, xgb_model, tf_model, tokenizer)

    demonette_nouns, tlfi_recursive_nouns = get_nouns_to_pipeline(
        "../data/wikidata/wikidata.db",
        "../data/wiktionary/wiktionary.db",
        "../data/nhuma/nhuma.csv",
        "../tlfi_scraping/words/words.json",
        "../tlfi_scraping/dbs/db_tlfi_recursive.db",
        "../demonette/demonette.csv",
    )

    if args.dataset == "tlfi_recursive":
        tlfi_recursive_nouns = list(tlfi_recursive_nouns)
        results = pipeline.process_words(tlfi_recursive_nouns)
    elif args.dataset == "demonette":
        demonette_nouns = list(demonette_nouns)
        results = pipeline.process_words(demonette_nouns)

    all_positive_df = get_all_positive_df(results, args.sum_prob)
    disagreement_df = get_disagreement_df(results, args.disagreement_type)

    all_positive_df.to_csv(args.positive_df_path)
    disagreement_df.to_csv(args.disagreement_df_path)
