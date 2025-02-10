import pandas as pd


def get_disagreement_df(
    results: dict[str, pd.DataFrame], disagreement_type: str = "two_pos_one_neg"
) -> pd.DataFrame:
    if disagreement_type not in ["two_pos_one_neg", "two_neg_one_pos"]:
        raise ValueError(
            "disagreement_type must be 'two_pos_one_neg' or 'two_neg_one_pos'"
        )

    combined_preds = pd.DataFrame(
        {
            "word": results["transformer"]["word"],
            "Transformer": results["transformer"]["prediction"],
            "logistic_regression": results["logistic_regression"]["prediction"],
            "XGBoost": results["xgboost"]["prediction"],
        }
    )

    # Combine probabilities and calculate the sum
    combined_probs = pd.DataFrame(
        {
            "word": results["transformer"]["word"],
            "Transformer_prob": results["transformer"]["probability"],
            "logistic_regression_prob": results["logistic_regression"]["probability"],
            "XGBoost_prob": results["xgboost"]["probability"],
        }
    )
    combined_probs["sum_prob"] = combined_probs[
        ["Transformer_prob", "logistic_regression_prob", "XGBoost_prob"]
    ].sum(axis=1)

    print(f"Checking disagreement with type {disagreement_type}")

    # Find disagreements with probabilities included
    disagreements = []
    words = []

    for idx, row in combined_preds.iterrows():
        models = ["Transformer", "logistic_regression", "XGBoost"]
        preds = [row[model] for model in models]

        # Check for disagreements based on predictions
        if sum(preds) == 2 and disagreement_type == "two_pos_one_neg":
            disagreements.append(
                preds
                + combined_probs.loc[
                    idx,
                    ["Transformer_prob", "logistic_regression_prob", "XGBoost_prob"],
                ].tolist()
                + [combined_probs.loc[idx, "sum_prob"]]
            )
            words.append(row["word"])
        elif sum(preds) == 1 and disagreement_type == "two_neg_one_pos":
            disagreements.append(
                preds
                + combined_probs.loc[
                    idx,
                    ["Transformer_prob", "logistic_regression_prob", "XGBoost_prob"],
                ].tolist()
                + [combined_probs.loc[idx, "sum_prob"]]
            )
            words.append(row["word"])

    if len(disagreements) > 0:
        columns = models + [f"{model}_prob" for model in models] + ["sum_prob"]
        return pd.DataFrame(disagreements, columns=columns, index=words)
    else:
        print("Found no disagreements")
        return pd.DataFrame()


def get_all_positive_df(
    results: dict[str, pd.DataFrame], sum_prob: int
) -> pd.DataFrame:
    # combine predictions from all models
    combined_preds = pd.DataFrame(
        {
            "word": results["transformer"]["word"],
            "Transformer": results["transformer"]["prediction"],
            "logistic_regression": results["logistic_regression"]["prediction"],
            "XGBoost": results["xgboost"]["prediction"],
        }
    )

    # find words where all models predicted positive
    all_positive_words = combined_preds[
        (combined_preds["Transformer"] == 1)
        & (combined_preds["logistic_regression"] == 1)
        & (combined_preds["XGBoost"] == 1)
    ]["word"].tolist()

    combined_probs = pd.DataFrame(
        {
            "word": results["transformer"]["word"],
            "Transformer_prob": results["transformer"]["probability"],
            "logistic_regression_prob": results["logistic_regression"]["probability"],
            "XGBoost_prob": results["xgboost"]["probability"],
        }
    )
    combined_probs["sum_prob"] = combined_probs[
        ["Transformer_prob", "logistic_regression_prob", "XGBoost_prob"]
    ].sum(axis=1)

    filtered_words = combined_probs[
        (combined_probs["word"].isin(all_positive_words))
        & (combined_probs["sum_prob"] >= sum_prob)
    ]

    return pd.DataFrame(
        {"word": filtered_words["word"], "sum_prob": filtered_words["sum_prob"]}
    ).reset_index()
