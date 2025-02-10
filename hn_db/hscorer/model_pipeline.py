import pandas as pd
import torch
import xgboost as xgb
from features import BaseMLClassifier


class ModelPipeline:
    def __init__(self, lr_model, xgb_model, transformer_model, tokenizer):
        self.lr_model = lr_model
        self.xgb_model = xgb_model
        self.transformer_model = transformer_model
        self.tokenizer = tokenizer

    def predict_lr(self, features) -> tuple[list[int], list[float]]:
        """Predict using logistic regression model"""
        predictions = self.lr_model.predict(features)
        probabilities = self.lr_model.predict_proba(features)[:, 1]
        return predictions, probabilities

    def predict_xgboost(self, features) -> tuple[list[int], list[float]]:
        """Predict using XGBoost model"""
        dmatrix = xgb.DMatrix(features)
        probabilities = self.xgb_model.predict(dmatrix)
        predictions = [1 if p >= 0.5 else 0 for p in probabilities]
        return predictions, probabilities

    def predict_transformer(self, words: list[str]) -> tuple[list[int], list[float]]:
        """Predict using Transformer model"""
        predictions = []
        probabilities = []

        for word in words:
            inputs = self.tokenizer(word, return_tensors="pt")

            with torch.no_grad():
                outputs = self.transformer_model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()

            predictions.append(pred)
            probabilities.append(probs[0][1].item())

        return predictions, probabilities

    def process_words(self, words: list[str]) -> dict[str, pd.DataFrame]:
        results = {}

        baseMLClassifier = BaseMLClassifier()
        features = baseMLClassifier.prepare_features(words)

        lr_preds, lr_probs = self.predict_lr(features)
        xgb_preds, xgb_probs = self.predict_xgboost(features)
        trans_preds, trans_probs = self.predict_transformer(words)

        results["logistic_regression"] = pd.DataFrame(
            {"word": words, "prediction": lr_preds, "probability": lr_probs}
        )

        results["xgboost"] = pd.DataFrame(
            {"word": words, "prediction": xgb_preds, "probability": xgb_probs}
        )

        results["transformer"] = pd.DataFrame(
            {"word": words, "prediction": trans_preds, "probability": trans_probs}
        )

        return results
