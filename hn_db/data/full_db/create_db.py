import sqlite3
import pandas as pd
import time


def create_database(dfs, col_names, db_path="hn_fr.db"):
    conn = sqlite3.connect(db_path)

    # create a unified set of all words
    all_words = set()
    for df, col in zip(dfs, col_names):
        all_words.update(df[col].dropna().unique())

    words_df = pd.DataFrame(
        {"word_id": range(1, len(all_words) + 1), "word": list(all_words)}
    )

    words_df["wikid_id"] = None
    words_df["wikt_id"] = None
    words_df["nhuma_id"] = None
    words_df["tlfi_id"] = None
    words_df["dmnt_id"] = None
    words_df["ncoll_id"] = None

    words_df.to_sql("words", conn, index=False, if_exists="replace")

    table_names = ["wikid", "wikt", "nhuma", "tlfi", "dmnt", "ncoll"]

    start_time = time.time()

    for df, col, table_name in zip(dfs, col_names, table_names):
        df = df.reset_index(drop=True)
        df.insert(0, f"{table_name}_id", df.index + 1)
        # df[f'{table_name}_id'] = df.index + 1

        df.to_sql(table_name, conn, index=False, if_exists="replace")

        cursor = conn.cursor()
        update_query = f"""
        UPDATE words
        SET {table_name}_id = (
            SELECT {table_name}_id
            FROM {table_name}
            WHERE {col} = words.word
        )
        """
        cursor.execute(update_query)

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX idx_words_word ON words(word)")

    for table_name, col in zip(table_names, col_names):
        cursor.execute(f"CREATE INDEX idx_{table_name}_word ON {table_name}({col})")

    conn.commit()
    conn.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Database creation completed in {elapsed_time:.2f} seconds.")
