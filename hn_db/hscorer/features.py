from hscorer import HScorer
from load_ft import model
from typing import List
import numpy as np


class BaseMLClassifier:
    """Base class containing common functionality for feature preparation"""

    def __init__(self):
        self.scorer = HScorer()

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get word embedding vector for a given word"""
        return model.get_word_vector(word)

    def prepare_features(self, words: List[str]) -> np.ndarray:
        """Extract features from a list of words, including word vectors"""
        features_list = []

        print(f"Preparing features for {len(words)} words...")

        for i, word in enumerate(words):
            if i % 10000 == 0:
                print(f"Processing word {i}/{len(words)}")

            # get scores (features)
            hypernym_scores = self.scorer.get_wordnet_hypernym_score(word)
            def_scores = self.scorer.get_wordnet_def_score(word)
            ft_scores = self.scorer.get_fasttext_score(word)
            suffix_score = self.scorer.get_suffix_score(word)

            if np.isnan(hypernym_scores).any():
                raise ValueError(
                    f"NaN found in hypernym_scores for word: {word}, "
                    f"hypernym_scores: {hypernym_scores}"
                )
            if np.isnan(def_scores).any():
                raise ValueError(
                    f"NaN found in def_scores for word: {word}, "
                    f"def_scores: {def_scores}"
                )
            if np.isnan(ft_scores).any():
                raise ValueError(
                    f"NaN found in ft_scores for word: {word}, "
                    f"ft_scores: {ft_scores}"
                )
            if np.isnan(suffix_score):
                raise ValueError(
                    f"NaN found in suffix_score for word: {word}, "
                    f"suffix_score: {suffix_score}"
                )

            word_vector = self.get_word_vector(word)

            if np.isnan(word_vector).any():
                raise ValueError(
                    f"NaN found in word_vector for word: {word}, "
                    f"word_vector: {word_vector}"
                )

            # combine all features
            features = np.concatenate(
                [hypernym_scores, def_scores, ft_scores, [suffix_score], word_vector]
            )

            features_list.append(features)

        return np.array(features_list)
