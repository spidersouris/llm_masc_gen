from collections import defaultdict
from typing import Any, TypeAlias
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# from filtering import nlp_main
import re
import argparse
import json

Thread: TypeAlias = list[dict[str, Any]]


def get_oasst2():
    # from https://github.com/keishihara/finetuning_llama3_hf/blob/3189dd6e40a174cc4ff5a176ebb15554a48befa2/src/tuner/datasets/oasst2_dataset_sequence.py
    def get_oasst2_dataset_sequence(
        lang: str = "fr",
    ) -> pd.DataFrame:
        """
        Returns tokenized oasst2 dataset sequence.

        Args:
            dataset_config (DatasetConfig)
            tokenizer (AutoTokenizer)
            format_fn (Callable[..., Chat]): A function that takes strings and returns single formated prompt.
            model_max_length (int, defaults to `2048`): Max token length that model can handle including prompt and response.
            split (str, defaults to `train`): Which split to return.

        Returns:
            ChatDataset: preprocessed dataset of specified split

        Examples:
            >>> from transformers import AutoTokenizer
            >>> from tuner.data import prepare_dataset

            >>> # instantiate tokenizer
            >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
            >>> tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            >>> tokenizer.eos_token = '<|eot_id|>'

            >>> # get the dataset
            >>> ds = prepare_dataset(
            >>>     'oasst2_dataset_sequence',
            >>>     tokenizer=tokenizer,
            >>>     model_max_length=8192,
            >>>     split='train,eval', # or 'train+eval'
            >>> )
            >>> train_ds, _ = ds['train'], ds['eval']
            >>> train_ds
            ChatDataset(name=chat_dataset:train, length=13423, shuffle=False, augmentation=False, truncation_side=right)
            >>> tokenizer.decode(train_ds[0]['input_ids']) # input tensors
            >>> tokenizer.decode(train_ds[0]['labels'][train_ds[0]['labels'] != -100]) # labels
        """
        rows = []

        ds = load_dataset("OpenAssistant/oasst2", split="train+validation")
        ds = ds.filter(lambda x: x["lang"] == lang)

        df = ds.to_pandas()
        df.role = df.role.replace({"prompter": "user"})
        df = df.rename(columns={"text": "content"})

        # prepare nodes and data_dict
        nodes = defaultdict(list)
        progbar = tqdm(
            df.to_dict(orient="records"),
            dynamic_ncols=True,
            desc="Building conversation tree…",
        )
        for data in progbar:
            if data["parent_id"]:
                nodes[data["parent_id"]].append(data["message_id"])
        nodes = dict(nodes)

        data_dict = df.set_index("message_id").transpose().to_dict()

        # restore all message threads
        def follow(thread: Thread, current_id: str) -> list[Thread]:
            # Given a thread and a current_id, return the thread that follows the current_id
            thread = [*thread, {"message_id": current_id, **data_dict[current_id]}]
            if current_id in nodes:
                new_thread = []
                for child_id in nodes[current_id]:
                    new_thread += follow(thread, child_id)
                return new_thread
            return [thread]

        def get_threads_from_root(root_id: str) -> list[Thread]:
            # Given a root_id, return all the threads in the tree
            all_threads = []
            thread = [
                {
                    "message_id": root_id,
                    **data_dict[root_id],
                }
            ]
            for child_id in nodes.get(root_id, []):
                all_threads += follow(thread, child_id)
            return all_threads

        df_filtered = df.copy()
        df_filtered = df_filtered[df_filtered.parent_id.isna()]

        tqdm.pandas(desc="Gathering threads…", dynamic_ncols=True)
        # make threads from root
        ser_thread = df_filtered.message_id.progress_apply(get_threads_from_root)
        ser_thread = ser_thread.explode().reset_index(drop=True)
        ser_thread = ser_thread[ser_thread.notna()]

        def cut_last_prompter_message(thread: Thread) -> Thread:
            if thread[-1]["role"] == "user":
                return thread[:-1]
            return thread

        tqdm.pandas(desc="Cutting last user message…", dynamic_ncols=True)
        ser_thread = ser_thread.progress_apply(cut_last_prompter_message)

        df_chat = ser_thread.to_frame(name="thread")

        def to_chat(thread: list[dict]):
            return [{k: t[k] for k in ["role", "content"]} for t in thread]

        df_chat["chat"] = df_chat.thread.apply(to_chat)

        for conversation in df_chat["chat"]:
            row = {}
            user_count = 1
            assistant_count = 1
            for message in conversation:
                if message["role"] == "user":
                    row[f"user_content{user_count}"] = message["content"]
                    user_count += 1
                elif message["role"] == "assistant":
                    row[f"assistant_content{assistant_count}"] = message["content"]
                    assistant_count += 1
            rows.append(row)

        return pd.DataFrame(rows).fillna("")

    oasst2_ds = get_oasst2_dataset_sequence().drop_duplicates(ignore_index=True)

    rows = []
    for conversation in oasst2_ds["chat"]:
        row = {}
        user_count = 1
        assistant_count = 1
        for message in conversation:
            if message["role"] == "user":
                row[f"user_content{user_count}"] = message["content"]
                user_count += 1
            elif message["role"] == "assistant":
                row[f"assistant_content{assistant_count}"] = message["content"]
                assistant_count += 1
        rows.append(row)

    rows.pop()  # remove last unproperly formatted row
    oasst2_df = pd.DataFrame(rows).fillna("")
    oasst2_df.to_pickle("dfs/oasst2/oasst2_df.pkl")


def get_oracle():
    oracle_jargon = ["oracle", "pythie", "oracles", "pythies"]

    def alpaca_to_df(json_file, keep_cols):
        with open(json_file, "r") as f:
            data = json.load(f)

        df = pd.DataFrame.from_dict(data)
        df = df[keep_cols]
        return df

    def remove_oracle_sents(text):
        if not any(term in text.lower() for term in oracle_jargon):
            return text

        doc = nlp_main(text)
        sentences = list(doc.sents)

        rx = re.compile(r"\b(?:" + "|".join(oracle_jargon) + r")\b", re.IGNORECASE)
        filtered_sentences = [
            sent.text for sent in sentences if not rx.search(sent.text)
        ]
        return " ".join(filtered_sentences)

    oracle_df = alpaca_to_df(
        "instruct_fr_wikipedia_oracle.json", ["instruction", "output"]
    )
    for column in oracle_df.columns:
        tqdm.pandas(desc=f"Processing {column}")
        oracle_df[column] = oracle_df[column].progress_apply(
            lambda text: remove_oracle_sents(text)
        )
    oracle_df.to_pickle("dfs/oracle/oracle_df.pkl")


def get_french_hh_rlhf():
    def ds_to_df(
        ds, ds_col: str, filter_by: str | None = None, unique_col: bool = False
    ):
        if filter_by not in [None, "user", "assistant"]:
            raise ValueError("filter_by must be None, 'user', or 'assistant'")

        rows = []
        for conversation in ds[ds_col]:
            row = {}
            user_count = 1
            assistant_count = 1
            for message in conversation:
                if message["role"] == "user":
                    if filter_by in [None, "user"]:
                        if (
                            unique_col and user_count == 1
                        ):  # Only take the first user message if unique_col is True
                            row["user_content"] = message["content"]
                            break
                        elif not unique_col:
                            row[f"user_content{user_count}"] = message["content"]
                            user_count += 1
                elif message["role"] == "assistant":
                    if filter_by in [None, "assistant"]:
                        if (
                            unique_col and assistant_count == 1
                        ):  # Only take the first assistant message if unique_col is True
                            row["assistant_content"] = message["content"]
                            break
                        elif not unique_col:
                            row[f"assistant_content{assistant_count}"] = message[
                                "content"
                            ]
                            assistant_count += 1
            rows.append(row)

        return pd.DataFrame(rows).fillna("")

    ds = load_dataset("AIffl/french_hh_rlhf", split="train+test")
    category = "helpful_base"
    hh_rlhf_ds = ds.filter(lambda x: x["category"] == category)
    hh_rlhf_df = ds_to_df(hh_rlhf_ds, "chosen", filter_by="user", unique_col=True)
    hh_rlhf_df.to_pickle("dfs/hh_rlhf/hh_rlhf_df.pkl")


def get_alpaca():
    ds = load_dataset("jpacifico/French-Alpaca-dataset-Instruct-55K", split="train")
    alpaca_instruct_df = pd.DataFrame(
        {
            "user_content": [
                row["instruction"] for row in ds if row["input"] in ["", "Aucun"]
            ]
        }
    )
    alpaca_instruct_df.to_pickle("dfs/alpaca/alpaca_df.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get instruction datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to get",
        choices=["oasst2", "oracle", "hh_rlhf", "alpaca"],
    )

    args = parser.parse_args()

    if args.dataset == "oasst2":
        get_oasst2()
    elif args.dataset == "oracle":
        get_oracle()
    elif args.dataset == "hh_rlhf":
        get_french_hh_rlhf()
    elif args.dataset == "alpaca":
        get_alpaca()
