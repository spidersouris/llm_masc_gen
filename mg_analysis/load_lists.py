import pandas as pd
import sqlite3


def get_human_list():
    return list(pd.read_pickle("dfs/human_df.pkl")["noun"])


def get_masc_gen_list():
    return list(pd.read_pickle("dfs/masc_gen_df.pkl")["noun"])


def get_neutrals_list():
    return ["personne", "personnes", "individu", "individus", "gens"]


def get_epicene_list():
    epicenes = ["médecin"]

    conn = sqlite3.connect("../hn_db/data/full_db/hn_fr.db")

    dmnt_df = pd.read_sql_query("SELECT * FROM dmnt", conn)
    # keep demonyms capitalized
    dmnt_df.update(
        dmnt_df.select_dtypes(include="object").apply(
            lambda col: col.where(dmnt_df["__dmnt__proper_noun"] == 1, col.str.lower())
        )
    )

    epicene_list = dmnt_df.loc[
        (dmnt_df["__dmnt__epicene"] == 1) & (dmnt_df["__dmnt__proper_noun"] == 0),
        "__dmnt__nm",
    ].tolist()

    conn.close()

    return epicenes + epicene_list
