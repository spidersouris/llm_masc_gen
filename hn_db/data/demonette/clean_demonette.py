import sqlite3

import pandas as pd


def check_duplicates(df, row):
    for i in range(1, 4):
        if row.name + i < len(df):
            if (
                row["nom"] in df.loc[row.name + i, "nom(s)_du_genre_opposé"]
                and df.loc[row.name + i, "nom"] in row["nom(s)_du_genre_opposé"]
            ):
                return True
    return False


def remove_duplicates(csv_file):
    df = pd.read_csv(csv_file, sep="\t")

    # df["is_duplicate"] = df.apply(lambda row: check_duplicates(df, row), axis=1)

    # df = df[~df["is_duplicate"]].drop(columns=["is_duplicate"])

    # rename column "nom" to "nm", and "nom(s)_du_genre_opposé" to "nf"
    df = df.rename(columns={"nom": "nm", "nom(s)_du_genre_opposé": "nf"})

    df["epicene"] = df.apply(lambda row: 1 if row["nm"] == row["nf"] else 0, axis=1)

    df["proper_noun"] = df.apply(lambda row: 1 if row["nm"][0].isupper() else 0, axis=1)

    # df["m_suffix"] = df["nm"].apply(
    #     lambda x: any(x.endswith(suffix) for suffix in suffixes["m"])
    # )

    # df["f_suffix"] = df["nf"].apply(
    #     lambda x: any(x.endswith(suffix) for suffix in suffixes["f"])
    # )
    return df


demonette_file = "demonette_full.csv"
df_og = remove_duplicates(demonette_file)

# keep nouns that are in human_db_base.db
conn = sqlite3.connect("human_db_base.db")
c = conn.cursor()
c.execute("SELECT DISTINCT word FROM words")
db_nouns = {row[0] for row in c.fetchall()}
conn.close()

df_filtered = df_og[
    df_og["nm"].isin(db_nouns) | df_og["nf"].isin(db_nouns)
].reset_index(drop=True)

# in df_filtered["nf"], replace ";" with "|"
df_filtered["nf"] = df_filtered["nf"].str.replace(";", "|")


def remove_nm_from_nf(row):
    if "|" in row["nf"]:
        print([word for word in row["nf"].split("|")], row["nm"])
        return "|".join([word for word in row["nf"].split("|") if word != row["nm"]])
    else:
        return row["nf"]


df_filtered["nf"] = df_filtered.apply(remove_nm_from_nf, axis=1)

print(len(df_filtered))

# check if any words in dmnt_pos.csv are not in human_db_base.db
dmnt_pos = pd.read_csv("dmnt_pos.csv")["word"]
dmnt_pos = dmnt_pos.to_frame().rename(columns={"word": "nm"})

# merge the two dataframes
df = pd.concat([df_filtered, dmnt_pos]).drop_duplicates().reset_index(drop=True)

print(f"Added {len(df) - len(df_filtered)} words from dmnt_pos.csv")

print(df.head())

df_merged = df.merge(df_og, on="nm", how="left", suffixes=("", "_df2"))

# update only the columns in df1 that have NaN values,
# using values from the corresponding columns in df2
for col in ["nf", "epicene", "proper_noun"]:
    df_merged[col] = df_merged[col].combine_first(df_merged[f"{col}_df2"])

# drop the temporary columns from df2
df_merged.drop(
    columns=[f"{col}_df2" for col in ["nf", "epicene", "proper_noun", "catégorie"]],
    inplace=True,
)

df_merged.drop(columns=["catégorie"], inplace=True)

print(df_merged)

df_merged.to_csv("dmnt.csv", sep="\t", index=False)
