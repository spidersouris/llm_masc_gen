import pandas as pd
import os
import numpy as np
from filtering import nlp_main, has_neutral


def load_filtered_df(name: str) -> pd.DataFrame:
    if name not in ["alpaca", "hh_rlhf", "oasst2", "oracle"]:
        raise ValueError(
            "name must be either 'alpaca', 'hh_rlhf', 'oasst2', or 'oracle'"
        )

    return pd.read_pickle(f"dfs/{name}/{name}_filtered_df.pkl")


def load_instructions(
    create_pkl: bool = False, small_sample: bool = True, test_sample: bool = False
) -> pd.DataFrame:
    if not create_pkl and not test_sample:
        if not os.path.exists("dfs/instructions_llm_inference.pkl"):
            all_instr = pd.read_pickle("dfs/instructions_llm_inference.pkl")
        else:
            all_instr = pd.read_pickle("dfs/instructions_llm_inference.pkl")
        print(
            f"Loaded {len(all_instr)} instructions from instructions_llm_inference.pkl"
        )

        if small_sample:
            print("Creating small sample")
            return create_sample(all_instr)

    print("Creating instructions_llm_inference.pkl")

    def is_instr_neutral(row: pd.Series) -> bool:
        doc = nlp_main(row["instruction"])
        return has_neutral(doc)

    oasst2_filtered_df = load_filtered_df("oasst2")
    oasst2_instr = pd.DataFrame()
    oasst2_instr["instruction"] = (
        oasst2_filtered_df["user_content1"].drop_duplicates().dropna()
    )
    oasst2_instr["source"] = "oasst2"
    oasst2_instr["neutral"] = oasst2_instr.apply(
        lambda row: is_instr_neutral(row), axis=1
    )

    oracle_filtered_df = load_filtered_df("oracle")
    oracle_instr = pd.DataFrame()
    oracle_instr["instruction"] = (
        oracle_filtered_df["instruction"].drop_duplicates().dropna()
    )
    oracle_instr["source"] = "oracle"
    oracle_instr["neutral"] = oracle_instr.apply(
        lambda row: is_instr_neutral(row), axis=1
    )

    hh_rlhf_filtered_df = load_filtered_df("hh_rlhf")
    hh_rlhf_instr = pd.DataFrame()
    hh_rlhf_instr["instruction"] = (
        hh_rlhf_filtered_df["user_content"].drop_duplicates().dropna()
    )
    hh_rlhf_instr["source"] = "hh_rlhf"
    hh_rlhf_instr["neutral"] = hh_rlhf_filtered_df["neutral"]

    alpaca_filtered_df = load_filtered_df("alpaca")
    alpaca_instr = pd.DataFrame()
    alpaca_instr["instruction"] = (
        alpaca_filtered_df["user_content"].drop_duplicates().dropna()
    )
    alpaca_instr["source"] = "alpaca"
    alpaca_instr["neutral"] = alpaca_filtered_df["neutral"]

    all_instr = pd.concat(
        [oasst2_instr, oracle_instr, hh_rlhf_instr, alpaca_instr], ignore_index=True
    ).drop_duplicates(ignore_index=True)

    # print(f"Length of oasst2 instructions: {len(oasst2_instr)}")
    # print(f"Length of oracle instructions: {len(oracle_instr)}")
    # print(f"Length of hh_rlhf instructions: {len(hh_rlhf_instr)}")
    # print(f"Length of alpaca instructions: {len(alpaca_instr)}")
    print(f"Loaded {len(all_instr)} instructions")

    if test_sample:
        return all_instr.sample(n=10, random_state=42)

    return all_instr


def create_sample(all_instr, sample_size: int = 10000):
    print(len(all_instr))
    all_instr = all_instr.replace("", np.nan, inplace=False).dropna()
    print(len(all_instr))
    source_lengths = all_instr.groupby("source").size()
    total_length = source_lengths.sum()
    source_proportions = source_lengths / total_length

    sample_sizes = (source_proportions * sample_size).astype(int)
    remainder = sample_size - sample_sizes.sum()

    # distribute the remainder across sources with the largest proportions
    if remainder > 0:
        sorted_sources = source_proportions.sort_values(ascending=False).index
        for source in sorted_sources[:remainder]:
            sample_sizes[source] += 1

    sampled_instr = pd.DataFrame()
    for source, source_sample_size in sample_sizes.items():
        source_df = all_instr[all_instr["source"] == source]
        source_sample = source_df.sample(
            n=min(source_sample_size, len(source_df)), random_state=42
        )
        sampled_instr = pd.concat([sampled_instr, source_sample], ignore_index=True)

    source_counts = sampled_instr["source"].value_counts()
    print("\nSampled entries per source:")
    print(source_counts)

    print(f"Sampled {len(sampled_instr)} instructions")
    return sampled_instr
