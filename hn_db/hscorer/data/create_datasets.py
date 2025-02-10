import pandas as pd
import numpy as np
import sqlite3
import argparse
from nltk.corpus import wordnet as wn


def create_human_dataset(
    wikidata_db_path="wikidata.db",
    wiktionary_db_path="wiktionary_final.db",
    nhuma_csv_path="nhuma.csv",
):
    # connect to the .db file and load the data from the 'wikidata' table
    conn = sqlite3.connect(wikidata_db_path)
    query = "SELECT entity_male_label, entityLabel FROM wikidata"
    wikidata_df = pd.read_sql_query(query, conn)
    conn.close()

    # same wiktionary
    conn = sqlite3.connect(wiktionary_db_path)
    query = "SELECT word FROM words"
    wiktionary_df = pd.read_sql_query(query, conn)
    conn.close()

    # load nhuma.csv and get first column
    csv_df = pd.read_csv(nhuma_csv_path)
    csv_first_column = csv_df.iloc[:, 0]

    # create noun column from wikidata data
    # replace empty entity_male_label rows with np.nan to allow replacement w/ entityLabel
    wikidata_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    wikidata_df["noun"] = wikidata_df.apply(
        lambda row: (
            row["entity_male_label"]
            if pd.notna(row["entity_male_label"])
            else row["entityLabel"].split()[0]
        ),
        axis=1,
    )

    # wikidata_df = wikidata_df[~wikidata_df['noun'].str.match(r'^[A-Z]')]

    # wiktionary_df = wiktionary_df[~wiktionary_df['word'].str.match(r'^[A-Z]|\(|·')]

    # remove words containing '(' or '·'
    wiktionary_df = wiktionary_df[~wiktionary_df["word"].str.contains(r"[\(·]")]

    # for words starting with uppercase (demonyms), randomly remove 90% for noise reduct.
    uppercase_df = wiktionary_df[wiktionary_df["word"].str.match(r"^[A-Z]")]
    lowercase_df = wiktionary_df[~wiktionary_df["word"].str.match(r"^[A-Z]")]

    # keep only 10% of uppercase words
    uppercase_sample = uppercase_df.sample(frac=0.2, random_state=1)

    # concat the lowercase words with the sampled uppercase words
    wiktionary_df = pd.concat([lowercase_df, uppercase_sample]).reset_index(drop=True)

    # shuffle the df
    wiktionary_df = wiktionary_df.sample(frac=1).reset_index(drop=True)

    # append nouns form nhuma.csv to the wikidata nouns
    all_nouns = (
        pd.concat(
            [wikidata_df["noun"], csv_first_column, wiktionary_df["word"]],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # create the final DF with 'word' and 'label' columns
    golden_df = pd.DataFrame({"word": all_nouns, "label": 1})

    golden_df.to_pickle("pkl/human.pkl")

    print("Human dataset created and saved as 'human.pkl'.")


def is_human_related(synset):
    """
    Check if a synset is related to humans by examining its hypernym paths
    """
    human_related_synsets = {
        wn.synset("person.n.01"),
        wn.synset("human.n.01"),
        wn.synset("human_being.n.01"),
    }

    for hypernym_path in synset.hypernym_paths():
        if any(human_synset in hypernym_path for human_synset in human_related_synsets):
            return True
    return False


def get_non_human_nouns(min_length=3):
    def get_words(synset):
        """Retrieve words from a synset if they meet the min_length criteria."""
        return {
            lemma.name()
            for lemma in synset.lemmas(lang="fra")
            if len(lemma.name()) >= min_length
        }

    def belongs_to_category(synset, category_name):
        """Check if a synset or its hypernyms belongs to a specific category."""
        return any(
            category_name in [s.name() for s in path]
            for path in synset.hypernym_paths()
        )

    def clean_words(words):
        # remove words with numbers
        words = {word for word in words if not any(char.isdigit() for char in word)}
        # remove words with special chars
        words = {
            word
            for word in words
            if not any(not char.isalnum() and char != "_" for char in word)
        }
        return {word.replace("_", " ") for word in words}

    artifact_nouns, object_nouns, living_being_nouns = set(), set(), set()

    for synset in wn.all_synsets(pos=wn.NOUN, lang="fra"):
        print(synset)
        if is_human_related(synset):  # skip human-related nouns
            continue

        words = clean_words(get_words(synset))

        if belongs_to_category(synset, "artifact.n.01"):
            artifact_nouns.update(words)
        elif belongs_to_category(synset, "living_thing.n.01"):
            living_being_nouns.update(words)
        elif belongs_to_category(synset, "object.n.01"):
            object_nouns.update(words)

    return sorted(object_nouns), sorted(living_being_nouns), sorted(artifact_nouns)


def create_non_human_dataset(label, human_dataset):
    """
    Create a dataset with non-human examples only
    """
    object_nouns, living_being_nouns, artifact_nouns = get_non_human_nouns()

    print(len(object_nouns), len(living_being_nouns), len(artifact_nouns))

    if label not in ["int", "category"]:
        raise ValueError("label must be either 'int' or 'category'")

    if label == "category":
        df = pd.DataFrame(
            columns=["word", "label"],
            data=[(word, "object") for word in object_nouns]
            + [(word, "living_being") for word in living_being_nouns]
            + [(word, "artifact") for word in artifact_nouns],
        )
    elif label == "int":
        df = pd.DataFrame(
            columns=["word", "label"],
            data=[(word, 0) for word in object_nouns]
            + [(word, 0) for word in living_being_nouns]
            + [(word, 0) for word in artifact_nouns],
        )

    golden_human_nouns = human_dataset["word"]

    # filter df to keep only rows where "word" is not in golden_human_nouns
    df_filtered = df[~df["word"].isin(golden_human_nouns)]

    removed_entries = pd.concat([df, df_filtered]).drop_duplicates(keep=False)
    print(f"{len(removed_entries)} entries were removed")

    print("Removed Entries:\n")
    print(removed_entries)

    # shuffle the DataFrame
    df_filtered = df_filtered.sample(frac=1).reset_index(drop=True)

    df.to_pickle("pkl/nonhuman.pkl")

    print("Non-human dataset created and saved as 'nonhuman.pkl'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wikidata_db_path",
        type=str,
        default="wikidata.db",
        help="Path to the wikidata.db file",
    )
    parser.add_argument(
        "--wiktionary_db_path",
        type=str,
        default="wiktionary_final.db",
        help="Path to the wiktionary.db file",
    )
    parser.add_argument(
        "--nhuma_csv_path",
        type=str,
        default="nhuma.csv",
        help="Path to the nhuma.csv file",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="int",
        help="Label type for non-human dataset (int or category)",
    )
    args = parser.parse_args()

    if args.label not in ["int", "category"]:
        raise ValueError("label must be either 'int' or 'category'")

    create_human_dataset(
        wikidata_db_path=args.wikidata_db_path,
        wiktionary_db_path=args.wiktionary_db_path,
        nhuma_csv_path=args.nhuma_csv_path,
    )
    human_dataset = pd.read_pickle("pkl/human.pkl")
    create_non_human_dataset(args.label, human_dataset)
