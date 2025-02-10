import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn

from load_ft import model

SUFFIX_FILE = "data/suffixes.csv"


class HScorer:
    def __init__(self):
        # Human-related seed words
        # !!! Seed words are used to calculate fasttext embedding similarity
        self.human_seeds: set[str] = {
            "personne",
            "homme",
            "femme",
            "individu",
            "humain",
            "gens",
        }

        # Non-human seed words
        self.non_human_seeds: set[str] = {
            "objet",
            "chose",
            "machine",
            "animal",
            "plante",
            "substance",
            "outil",
            "appareil",
            "sentiment",
        }

        # Human indicators in definitions
        # !!! Indicators are used to calculate number of occurrences in WN definitions
        self.human_indicators: set[str] = {
            # Roles and occupations
            "someone",
            "somebody",
            "person",
            "people",
            "individual",
            "professional",
            "specialist",
            "expert",
            "practitioner",
            "worker",
            "employee",
            "occupation",
            "profession",
            # Personal characteristics
            "who",
            "whose",
            "personality",
            "character",
            # Social roles
            "member",
            "leader",
            "follower",
            "participant",
            # Family relations
            "parent",
            "child",
            "sibling",
            "relative",
        }

        # Non-human indicators in definitions
        self.non_human_indicators: set[str] = {
            # Objects and items
            "object",
            "thing",
            "item",
            "device",
            "tool",
            "machine",
            "equipment",
            "apparatus",
            "instrument",
            # Natural elements
            "substance",
            "material",
            "element",
            "chemical",
            # Living non-human
            "plant",
            "animal",
            "organism",
            "species",
            # Abstract concepts
            "concept",
            "idea",
            "notion",
            "theory",
            "principle",
            # Places and locations
            "place",
            "location",
            "area",
            "region",
            "structure",
            # Measurements and quantities
            "amount",
            "quantity",
            "measure",
            "unit",
            # States and conditions
            "state",
            "condition",
            "situation",
            "circumstance",
        }

    def get_wordnet_hypernym_score(self, word: str) -> tuple[float, float]:
        """
        Calculate both human and non-human scores based on WordNet hypernym paths
        Returns: (human_score, non_human_score)
        """
        human_score = 0
        non_human_score = 0
        synsets = wn.synsets(word, pos=wn.NOUN, lang="fra")

        if synsets is None:
            print(f"No synsets found for word: {word}")
            return 0.0, 0.0

        synsets = [s for s in synsets if s is not None]
        for synset in synsets:
            hypernym_paths = synset.hypernym_paths()
            for path in hypernym_paths:
                for hypernym in path:
                    hypernym_str = str(hypernym)

                    # check for human indicators
                    if any(
                        indicator in hypernym_str
                        for indicator in [
                            "person.n.01",
                            "human.n.01",
                            "worker.n.01",
                            "employee.n.01",
                        ]
                    ):
                        human_score += 1

                    # check for non-human indicators
                    if any(
                        indicator in hypernym_str
                        for indicator in [
                            # "object.n.01", # not good, too generic
                            "artifact.n.01",
                            "substance.n.01",
                            "plant.n.01",
                            "animal.n.01",
                        ]
                    ):
                        non_human_score += 1

        total_paths = len(synsets)
        return (
            human_score / total_paths if total_paths > 0 else 0,
            non_human_score / total_paths if total_paths > 0 else 0,
        )

    def get_wordnet_def_score(self, word: str) -> tuple[float, float]:
        """
        Calculate both human and non-human scores based on WordNet definition analysis
        Returns: (human_score, non_human_score)
        """
        human_score = 0
        non_human_score = 0
        synsets = wn.synsets(word, pos=wn.NOUN, lang="fra")

        if synsets is None:
            print(f"No synsets found for word: {word}")
            return 0.0, 0.0

        synsets = [s for s in synsets if s is not None]
        for synset in synsets:
            definition = synset.definition().lower()

            # count human indicators in definition
            human_indicators_found = sum(
                1 for indicator in self.human_indicators if indicator in definition
            )
            if human_indicators_found > 0:
                human_score += human_indicators_found

            # count non-human indicators in definition
            non_human_indicators_found = sum(
                1 for indicator in self.non_human_indicators if indicator in definition
            )
            if non_human_indicators_found > 0:
                non_human_score += non_human_indicators_found

        total_synsets = len(synsets)
        return (
            human_score / total_synsets if total_synsets > 0 else 0,
            non_human_score / total_synsets if total_synsets > 0 else 0,
        )

    def get_fasttext_score(self, word: str) -> tuple[float, float]:
        """
        Calculate both human and non-human scores based on FastText similarity
        Returns: (human_score, non_human_score)
        """
        try:
            word_vector = model.get_word_vector(word)

            if not word_vector.any():
                # all zeroes
                print(f"Word vector is all zeroes for word: {word}")
                return 0.0, 0.0

            # calculate similarities with human seeds
            human_cos = []
            for seed in self.human_seeds:
                seed_vector = model.get_word_vector(seed)
                cos_sim = (word_vector @ seed_vector.T) / (
                    np.linalg.norm(word_vector) * np.linalg.norm(seed_vector)
                )
                human_cos.append(cos_sim)

            # calculate similarities with non-human seeds
            non_human_cos = []
            for seed in self.non_human_seeds:
                seed_vector = model.get_word_vector(seed)
                cos_sim = (word_vector @ seed_vector.T) / (
                    np.linalg.norm(word_vector) * np.linalg.norm(seed_vector)
                )
                non_human_cos.append(cos_sim)

            human_score = sum(human_cos) / len(human_cos) if human_cos else 0
            non_human_score = (
                sum(non_human_cos) / len(non_human_cos) if non_human_cos else 0
            )

            return human_score, non_human_score

        except Exception as e:
            print(f"Error processing word '{word}': {e}")
            return 0.0, 0.0

    def get_suffix_score(self, word: str):
        """
        Calculate score based on word suffix
        """
        try:
            male_suffixes, female_suffixes = self.load_suffix_file(SUFFIX_FILE)

            if word.endswith(tuple(male_suffixes)):
                return 1
            elif word.endswith(tuple(female_suffixes)):
                return 1
            return 0
        except Exception as e:
            print(f"Error processing word '{word}': {e}")
            return 0

    def load_suffix_file(self, file_path: str):
        """
        Load a list of suffixes from a file
        """
        df = pd.read_csv(file_path, header=0, sep="\t")
        male_suffixes = df.values[:, 0].tolist()
        female_suffixes = df.values[:, 1].tolist()
        return male_suffixes, female_suffixes

    # this is for the RBS-only system (testing)
    # not used for model training
    def calculate_classification_score(
        self,
        word: str,
        weights: tuple[float, float, float, float] = (0.4, 0.3, 0.3, 0.3),
    ) -> dict:
        hypernym_human, hypernym_non_human = self.get_wordnet_hypernym_score(word)
        def_human, def_non_human = self.get_wordnet_def_score(word)
        ft_human, ft_non_human = self.get_fasttext_score(word)
        suffix_score = self.get_suffix_score(word)

        # weighted scores
        human_score = (
            hypernym_human * weights[0]
            + def_human * weights[1]
            + ft_human * weights[2]
            + suffix_score * weights[3]
        )
        non_human_score = (
            hypernym_non_human * weights[0]
            + def_non_human * weights[1]
            + ft_non_human * weights[2]
            + suffix_score * weights[3]
        )

        return {
            "classification": "human" if human_score > non_human_score else "non_human",
            "human_score": human_score,
            "non_human_score": non_human_score,
            "detailed_scores": {
                "hypernym": {"human": hypernym_human, "non_human": hypernym_non_human},
                "definition": {"human": def_human, "non_human": def_non_human},
                "fasttext": {"human": ft_human, "non_human": ft_non_human},
                "suffix": suffix_score,
            },
        }

    # RBS
    def classify_words(
        self, words: list[str], confidence_threshold: float = 0.02
    ) -> dict:
        results = {"human": [], "non_human": [], "uncertain": [], "scores": {}}

        for word in words:
            scores = self.calculate_classification_score(word)
            results["scores"][word] = scores

            human_score = scores["human_score"]
            non_human_score = scores["non_human_score"]
            score_difference = abs(human_score - non_human_score)

            if score_difference < confidence_threshold:
                results["uncertain"].append(word)
            elif human_score > non_human_score:
                results["human"].append(word)
            else:
                results["non_human"].append(word)

        return results
