import json
import re
import argparse


# EVAL VARS
# GPT_OUTPUT_FILES = ["gpt_eval_assistant_human_batch_gpto.jsonl"]
# GPT_OUTPUT_FILES = ["gpt_eval_assistant_human_batch_gpto_fs.jsonl"]
# GPT_OUTPUT_FILES = ["masc_gen_paper/eval/gpt_eval_classification-500.jsonl"]
# ORIGINAL_RESULTS_FILES = ["masc_gen_paper/eval/gpt_eval_df_assistant_results-500.json"]
# FINAL_FILES = ["masc_gen_paper/eval/gpt_eval_df_assistant_results-500_fs_final.json"]
def gpt_eval(gpt_output_file, original_results_file, final_file, positive_only):
    for i, (gpt_output_file, original_results_file, final_file) in enumerate(
        zip(GPT_OUTPUT_FILES, ORIGINAL_RESULTS_FILES, FINAL_FILES)
    ):
        print(
            f"Processing {gpt_output_file} | {original_results_file} | {final_file}\n\n"
        )

        failed_processed_count = 0

        # Load GPT output file
        with open(gpt_output_file, "r", encoding="utf-8") as f:
            gpt_outputs = {}
            for line in f:
                data = json.loads(line)
                if (
                    "response" in data
                    and "body" in data["response"]
                    and "choices" in data["response"]["body"]
                ):
                    id = data["custom_id"]
                    content = data["response"]["body"]["choices"][0]["message"][
                        "content"
                    ]
                    try:
                        parsed_content = json.loads(content)

                        filtered_content = (
                            {k: v for k, v in parsed_content.items() if v == 1}
                            if positive_only
                            else parsed_content
                        )
                        gpt_outputs[id] = filtered_content
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON for ID {id}: {content}")
                        failed_processed_count += 1

        # Load original results JSON file
        with open(original_results_file, "r", encoding="utf-8") as f:
            original_results = json.load(f)

        processed_count = 0
        for entry in original_results:
            id = entry["text_index_dataset"]
            if id in gpt_outputs:
                gpt_output = gpt_outputs[id]
                entry["real_human_nouns"] = gpt_output
                entry["real_human_nouns_count"] = len(entry["real_human_nouns"])

                real_masc_gen_logs = []
                for log in entry["masc_gen_logs"]:
                    token = log["token"]

                    # Match token with or without numeric suffix in real_human_nouns
                    pattern = re.compile(rf"^{re.escape(token)}(_\d+)?$")
                    if any(pattern.match(k) for k in gpt_output):
                        real_masc_gen_logs.append(log)

                entry["real_masc_gen_logs"] = real_masc_gen_logs

                processed_count += 1

        print(f"Processed entries: {processed_count}")
        print(f"Failed to process entries: {failed_processed_count}")

        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(original_results, f, indent=4, ensure_ascii=False)

        print(f"Successfully wrote {processed_count} entries to {final_file}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpt_output_files",
        nargs="+",
        help="List of GPT output files to process (jsonl)",
    )
    parser.add_argument(
        "--original_results_files",
        nargs="+",
        help="List of original results files to process (json)",
    )
    parser.add_argument(
        "--final_files",
        nargs="+",
        help="List of final files to write processed results to (json)",
    )
    parser.add_argument(
        "--positive_only",
        action="store_true",
        help="set to True to only include positive tokens in the final results. should be False for eval data generation",
    )
    args = parser.parse_args()

    GPT_OUTPUT_FILES = args.gpt_output_files
    ORIGINAL_RESULTS_FILES = args.original_results_files
    FINAL_FILES = args.final_files
    POSITIVE_ONLY = args.positive_only
    gpt_eval(GPT_OUTPUT_FILES, ORIGINAL_RESULTS_FILES, FINAL_FILES, POSITIVE_ONLY)
