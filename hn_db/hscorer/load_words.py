import json
import sqlite3
import pandas as pd
import numpy as np
import csv


def get_db_values(
    db_path: str,
    table_name: str,
    column_name: str,
    use_alternate_column: bool = False,
    alternate_column: str = "",
) -> list | None:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if use_alternate_column and alternate_column:
            query = (
                f"SELECT COALESCE({column_name}, {alternate_column}) FROM {table_name}"
            )
        else:
            query = f"SELECT {column_name} FROM {table_name}"
        cursor.execute(query)

        values = [row[0] for row in cursor.fetchall()]

        conn.close()

        return values

    except sqlite3.Error as e:
        print(f"Error accessing database {db_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error with database {db_path}: {str(e)}")
        return None


def get_csv_values(csv_path: str, column_name: str) -> list | None:
    try:
        with open(csv_path, "r") as file:
            reader = csv.DictReader(file)
            values = [row[column_name] for row in reader]

        return values

    except Exception as e:
        print(f"Unexpected error with CSV file {csv_path}: {str(e)}")
        return None


def get_json_values(json_path: str, column_name: str) -> list:
    with open(json_path, "r") as file:
        data = json.load(file)

        values = [row[column_name] for row in data]

    return values


def get_nouns_to_pipeline(
    wikidata_db_path: str,
    wiktionary_db_path: str,
    nhuma_csv_path: str,
    tlfi_json_path: str,
    tlfi_recursive_db_path: str,
    demonette_csv_path: str,
) -> tuple[set, set]:
    conn = sqlite3.connect(wikidata_db_path)
    query = "SELECT entity_male_label, entityLabel FROM wikidata"
    wikidata_df = pd.read_sql_query(query, conn)
    conn.close()

    conn = sqlite3.connect(wiktionary_db_path)
    query = "SELECT word FROM words"
    wiktionary_df = pd.read_sql_query(query, conn)
    conn.close()

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

    nhuma_df = pd.read_csv(nhuma_csv_path)

    # THOSE ARE GOLDEN NOUNS WE WANT TO USE TO FILTER TLFI_RECURSIVE/DEMONETTE
    tlfi_nouns = set(get_json_values(tlfi_json_path, "wordForm"))
    wiktionary_nouns = set(wiktionary_df["word"])
    wikidata_nouns = set(wikidata_df["noun"])
    nhuma_nouns = nhuma_df.iloc[:, 0]

    # THOSE ARE NOUNS WE WANT TO FILTER
    tlfi_recursive_nouns = set(
        get_db_values(tlfi_recursive_db_path, "words", "wordForm")
    )
    demonette_nouns = set(get_csv_values(demonette_csv_path, "nm"))

    print("Golden:")
    print(f"Got {len(tlfi_nouns)} nouns from TLFI")
    print(f"Got {len(wiktionary_nouns)} nouns from Wiktionary")
    print(f"Got {len(wikidata_nouns)} nouns from Wikidata")
    print(f"Got {len(nhuma_nouns)} nouns from NHUMA")

    print("To filter:")
    print(f"Got {len(tlfi_recursive_nouns)} nouns from TLFI recursive")
    print(f"Got {len(demonette_nouns)} nouns from Demonette")

    demonette_nouns = set([w for w in demonette_nouns if not w[0].isupper()])

    print(f"Got {len(demonette_nouns)} nouns from Demonette (filtered)")

    # all_nouns = tlfi_nouns.union(tlfi_recursive_nouns, demonette_nouns)
    exclude_nouns = tlfi_nouns.union(wiktionary_nouns, wikidata_nouns, nhuma_nouns)

    # remove dups
    demonette_nouns.difference_update(exclude_nouns)
    tlfi_recursive_nouns.difference_update(exclude_nouns)

    demonette_nouns.difference_update(tlfi_recursive_nouns)

    print(
        "Total number of non-duplicate nouns: ",
        f"{len(demonette_nouns) + len(tlfi_recursive_nouns)}",
    )

    return demonette_nouns, tlfi_recursive_nouns
