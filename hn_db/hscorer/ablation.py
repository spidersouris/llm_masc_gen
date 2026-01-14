import argparse
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from hscorer import HScorer
from load_ft import model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "--human_pkl_path", type=str, help="Path to the human dataset pickle file"
)
parser.add_argument(
    "--non_human_pkl_path", type=str, help="Path to the non-human dataset pickle file"
)
parser.add_argument(
    "--model",
    type=str,
    default="lr",
    choices=["lr", "xgboost"],
    help="Model to use for ablation study",
)
parser.add_argument(
    "--output",
    type=str,
    default="ablation_results.csv",
    help="Path to save ablation results CSV",
)
parser.add_argument(
    "--latex_table",
    action="store_true",
    help="Output results in LaTeX table format from CSV and exit",
)
args = parser.parse_args()


class AblationClassifier:
    def __init__(self, exclude_features=None):
        self.scorer = HScorer()
        self.exclude_features = exclude_features or []

    def get_word_vector(self, word: str) -> np.ndarray:
        return model.get_word_vector(word)

    def prepare_features(self, words):
        features_list = []
        print(f"Preparing features for {len(words)} words...")
        print(f"Excluding features: {self.exclude_features}")

        for i, word in enumerate(words):
            if i % 10000 == 0:
                print(f"Processing word {i}/{len(words)}")

            feature_components = []

            # WN Hypernym scores (2 features)
            if "H" not in self.exclude_features:
                hypernym_scores = self.scorer.get_wordnet_hypernym_score(word)
                feature_components.append(hypernym_scores)

            # WN Definition scores (2 features)
            if "D" not in self.exclude_features:
                def_scores = self.scorer.get_wordnet_def_score(word)
                feature_components.append(def_scores)

            # FastText similarity scores (2 features)
            if "F" not in self.exclude_features:
                ft_scores = self.scorer.get_fasttext_score(word)
                feature_components.append(ft_scores)

            # Suffix score (1 feature)
            if "S" not in self.exclude_features:
                suffix_score = self.scorer.get_suffix_score(word)
                feature_components.append([suffix_score])

            # Word embedding (300 features)
            if "W" not in self.exclude_features:
                word_vector = self.get_word_vector(word)
                feature_components.append(word_vector)

            if feature_components:
                features = np.concatenate(feature_components)
            else:
                raise ValueError("Cannot concatenate zero features.")

            features_list.append(features)

        return np.array(features_list)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="binary"),
    }

    return metrics


def train_eval_lr(X_train, y_train, X_test, y_test):
    # same parameters as best model found with HPO
    lr = LogisticRegression(penalty="l1", C=100, solver="saga", max_iter=1000)
    lr.fit(X_train, y_train)

    return evaluate_model(lr, X_test, y_test)


def train_eval_xgboost(X_train, y_train, X_test, y_test):
    # same parameters as best model found with HPO
    params = {
        "booster": "gbtree",
        "learning_rate": 0.22394632872649503,
        "max_depth": 10,
        "min_child_weight": 78,
        "subsample": 1,
        "colsample_bytree": 1,
        "n_estimators": 912,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 0,
        "objective": "binary:logistic",
        # "early_stopping_rounds": 20,
        "random_state": 42,
        "tree_method": "gpu_hist",
        "n_jobs": -1,
    }

    xgbmodel = xgb.XGBClassifier(**params)
    xgbmodel.fit(X_train, y_train)

    return evaluate_model(xgbmodel, X_test, y_test)


def run_ablation(
    human_pkl_path,
    non_human_pkl_path,
    model_type="lr",
    output_csv="ablation_results.csv",
):
    print(f"HUMAN DATASET PATH: {human_pkl_path}")
    print(f"NON-HUMAN DATASET PATH: {non_human_pkl_path}")
    print(f"MODEL TYPE: {model_type}")

    human_dataset = pd.read_pickle(human_pkl_path)
    non_human_dataset = pd.read_pickle(non_human_pkl_path)
    merged_dataset = pd.concat([non_human_dataset, human_dataset], ignore_index=True)

    X = merged_dataset["word"]
    y = merged_dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    feature_groups = [
        "H",  # WordNet Hypernym Score
        "D",  # WordNet Definition Score
        "F",  # FastText Similarity Score
        "S",  # Suffix Score
        "W",  # Word Embedding
    ]

    results = []

    train_eval_fn = {
        "lr": train_eval_lr,
        "xgboost": train_eval_xgboost,
    }[model_type]

    def run_experiment(experiment_name, exclude_features, header):
        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)

        clf = AblationClassifier(exclude_features=exclude_features)
        X_train_prep = clf.prepare_features(X_train)
        X_test_prep = clf.prepare_features(X_test)

        metrics = train_eval_fn(X_train_prep, y_train, X_test_prep, y_test)

        results.append({"experiment": experiment_name, **metrics})
        print(
            f"Results: Accuracy={metrics['accuracy']:.4f}, " f"F1={metrics['f1']:.4f}"
        )

    # baseline (all features)
    run_experiment(
        experiment_name="all",
        exclude_features=[],
        header="Training baseline model with ALL features",
    )

    # ablation (remove one feature)
    for feature in feature_groups:
        run_experiment(
            experiment_name=f"wo_{feature}",
            exclude_features=[feature],
            header=f"Training model WITHOUT: {feature}",
        )

    # only individual features
    for feature in feature_groups:
        run_experiment(
            experiment_name=f"only_{feature}",
            exclude_features=[f for f in feature_groups if f != feature],
            header=f"Training model with ONLY: {feature}",
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Ablation study complete; results saved to: {output_csv}")
    print(f"{'='*80}\n")

    # print(results_df.to_string(index=False))
    print(results_df.to_latex(index=False, escape=True))

    baseline_acc = results_df[results_df["experiment"] == "all"]["accuracy"].values[0]

    feature_importance_df = pd.DataFrame(columns=["feature", "acc_drop", "drop_pct"])

    print("\n\nFeature importance:")
    for feature in feature_groups:
        without_feature = results_df[results_df["experiment"] == f"wo_{feature}"][
            "accuracy"
        ].values[0]
        drop = baseline_acc - without_feature
        drop_pct = (drop / baseline_acc) * 100
        feature_importance_df.loc[len(feature_importance_df)] = {
            "feature": feature,
            "acc_drop": drop,
            "drop_pct": drop_pct,
        }

    feature_importance_df.to_csv(f"feature_importance_{model_type}.csv", index=False)

    # print(feature_importance_df.to_string(index=False))
    print(feature_importance_df.to_latex(index=False, escape=True))

    return results_df


if args.latex_table:
    lr_ablation_df = pd.read_csv("ablation_results_lr.csv")
    xgb_ablation_df = pd.read_csv("ablation_results_xgboost.csv")

    lr_ablation_df.rename(
        columns={
            "precision": "precision_lr",
            "recall": "recall_lr",
            "f1": "f1_lr",
            "accuracy": "accuracy_lr",
        },
        inplace=True,
    )
    xgb_ablation_df.rename(
        columns={
            "precision": "precision_xgb",
            "recall": "recall_xgb",
            "f1": "f1_xgb",
            "accuracy": "accuracy_xgb",
        },
        inplace=True,
    )

    abl_merged_df = pd.merge(lr_ablation_df, xgb_ablation_df, on="experiment")

    lr_fi_df = pd.read_csv("feature_importance_lr.csv")
    xgb_fi_df = pd.read_csv("feature_importance_xgboost.csv")

    lr_fi_df.rename(
        columns={
            "acc_drop": "acc_drop_lr",
            "drop_pct": "drop_pct_lr",
        },
        inplace=True,
    )
    xgb_fi_df.rename(
        columns={
            "acc_drop": "acc_drop_xgb",
            "drop_pct": "drop_pct_xgb",
        },
        inplace=True,
    )
    fi_merged_df = pd.merge(lr_fi_df, xgb_fi_df, on="feature")

    print("\nAblation:")
    print(abl_merged_df.to_latex(index=False, escape=True))

    print("\nFeature Importance:")
    print(fi_merged_df.to_latex(index=False, escape=True))
    sys.exit(0)

run_ablation(
    args.human_pkl_path,
    args.non_human_pkl_path,
    model_type=args.model,
    output_csv=f"ablation_results_{args.model}.csv",
)
