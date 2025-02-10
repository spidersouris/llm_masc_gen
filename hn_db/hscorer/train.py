import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from features import BaseMLClassifier


def train_lr(X_train, y_train, output="models/lr.pkl"):
    from sklearn.linear_model import LogisticRegression
    import joblib

    lr = LogisticRegression(penalty="l1", C=100, solver="saga")
    lr.fit(X_train, y_train)

    with open(output, "wb") as f:
        joblib.dump(lr, f)


def train_xgboost(X_train, y_train, X_test, y_test, output="models/xgb.json"):
    import xgboost as xgb

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
        "early_stopping_rounds": 20,
        "random_state": 42,
        "tree_method": "gpu_hist",
        "n_jobs": 24,
    }

    xgbmodel = xgb.XGBClassifier(**params)

    xgbmodel.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    xgbmodel.save_model(output)


def train_tf(X_train, y_train, X_test, y_test, output="models/tf"):
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    import evaluate
    import torch

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            "camembert-base",
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            problem_type="single_label_classification",
        )

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

    train_dataset = TextDataset(train_encodings, y_train.tolist())
    test_dataset = TextDataset(test_encodings, y_test.tolist())

    training_args = TrainingArguments(
        output_dir="hscorer",
        logging_steps=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        weight_decay=0.061748150962771656,
        learning_rate=8.497821083760116e-06,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    trainer.save_model(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models for human/non-human classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "human_pkl_path",
        default="data/pkl/human.pkl",
        type=str,
        help="Path to the human dataset pickle file",
    )
    parser.add_argument(
        "non_human_pkl_path",
        default="data/pkl/nonhuman.pkl",
        type=str,
        help="Path to the non-human dataset pickle file",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model to train (lr, xgboost, transformer)",
        choices=["lr", "xgboost", "transformer"],
    )
    parser.add_argument("output_file", type=str, help="Path to save the trained model")
    args = parser.parse_args()

    if args.model not in ["lr", "xgboost", "transformer"]:
        print(
            "Invalid model type. Please choose from 'lr', 'xgboost', or 'transformer'"
        )
        sys.exit(1)

    human_dataset = pd.read_pickle(args.human_pkl_path)
    non_human_dataset = pd.read_pickle(args.non_human_pkl_path)

    merged_human_non_human = pd.concat(
        [non_human_dataset, human_dataset], ignore_index=True
    )

    clf = BaseMLClassifier()

    X = merged_human_non_human["word"]
    y = merged_human_non_human["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    X_train = clf.prepare_features(X_train)
    X_test = clf.prepare_features(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if args.model == "lr":
        train_lr(X_train, y_train, args.output_file)
    elif args.model == "xgboost":
        train_xgboost(X_train, y_train, X_test, y_test, args.output_file)
    elif args.model == "transformer":
        train_tf(X_train, y_train, X_test, y_test, args.output_file)
