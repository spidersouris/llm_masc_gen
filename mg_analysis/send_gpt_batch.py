import argparse
import json
import os
from typing import Annotated, Literal

from openai import OpenAI
from pydantic import BeforeValidator, RootModel, ValidationError


def _as_label(v):
    return int(v) if isinstance(v, str) and v.strip() in ("0", "1") else v


class HNValidation(
    RootModel[dict[str, Annotated[Literal[0, 1], BeforeValidator(_as_label)]]]
):
    """Model to validate the output of the human noun classification."""


client = OpenAI()


def get_prompt(text, human_nouns, fs=True):
    prompt = """
    Given a text and nouns, for each noun, determine if it is a human noun in context.
    Some nouns may appear multiple times in the text. In such cases, they are distinguished by ID ("noun_1", "noun_2"…), following the order in which they appear.
    Do not assume that all occurrences of the same noun are either human or non-human; instead, assess each occurrence individually based on its unique context.
    Only respond in this format, where human_noun is the noun being considered.
    {{
      "human_noun": 0,
      "human_noun_2": 1
    }}
    """

    examples = """\n\n
    ## Examples
    Text: Les facteurs d'employabilité des facteurs, chargés de distribuer le courrier, vont évoluer.
    Nouns: facteurs, facteurs_2
    Output: {{ "facteurs": 0, "facteurs_2": 1 }}

    Text: Le président a annoncé aux citoyens une série de mesures pour renforcer l'économie du pays.
    Nouns: président, citoyens, mesures
    Output: {{ "président": 1, "citoyens": 1, "mesures": 0 }}

    Text: Il croit aux esprits et aux fantômes depuis qu'il est enfant.
    Nouns: esprits, fantômes, enfant
    Output: {{ "esprits": 0, "fantômes": 0, "enfant": 1 }}
    """

    to_analyze = f"""\n\n
    Text: {text}
    Nouns: {human_nouns}
    Output:
    """

    if fs:
        return prompt + examples + to_analyze

    return prompt + to_analyze


def format_batch(i, prompt, temp):
    return {
        "custom_id": f"{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful French assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 500,
            "temperature": temp,
        },
    }


def validate_output(batch_results):
    data = json.loads(batch_results)
    for item in data:
        try:
            HNValidation.model_validate_json(
                item["response"]["choices"][0]["message"]["content"]
            )
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Validation error for item {item['custom_id']}: {e}")


def write_batch_file(
    json_path,
    dataset,
    content_type,
    is_e2=False,
    is_local=False,
    batch=True,
    temp=1.0,
    max_results=None,
):
    with open(json_path, "r") as f:
        input_json = f.read()

    data = json.loads(input_json)
    llm_type = "prop" if not is_local else "local"

    for i, d in enumerate(data):
        if max_results is not None and i >= max_results:
            break

        if i % 100 == 0:
            print(f"Processing item {i+1}/{len(data)}")

        text = d["text"]
        human_nouns = list(d["human_nouns"])

        if len(human_nouns) == 0:
            continue

        if human_nouns != set(human_nouns):
            human_nouns_with_ids = []
            counts = {}
            for noun in human_nouns:
                counts[noun] = counts.get(noun, 0) + 1
                if counts[noun] > 1:
                    human_nouns_with_ids.append(f"{noun}_{counts[noun]}")
                else:
                    human_nouns_with_ids.append(noun)

        if human_nouns_with_ids:
            human_nouns = human_nouns_with_ids

        prompt = get_prompt(text, human_nouns)
        if batch:
            prompt = format_batch(d["text_index_dataset"], prompt, temp)
            # with open(f"masc_gen_paper/eval/{dataset}_{content_type}_human_batch.jsonl", "a") as f:
            os.makedirs(
                f"instr_outputs_mg_results/unreal/llm_{llm_type}/batches",
                exist_ok=True,
            )
            is_e2_label = "_e2" if is_e2 else "_e1"
            with open(
                f"instr_outputs_mg_results/unreal/llm_{llm_type}/batches/{dataset}_{content_type}{is_e2_label}_human_batch.jsonl",
                "a",
            ) as f:
                f.write(json.dumps(prompt) + "\n")


def upload_batch(batch_file):
    batch_input_file = client.files.create(file=open(batch_file, "rb"), purpose="batch")
    return batch_input_file


def send_batch(uploaded_batch_file, description, completion_window="24h"):
    batch_input_file_id = uploaded_batch_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,  # type: ignore
        metadata={
            "description": description,
        },
    )

    print(f"Batch {description} sent successfully!")


def get_batch_results(output_file_id, output_file_name=None, save_to_file=True):
    if not save_to_file:
        return client.files.content(output_file_id).text

    if output_file_name is None:
        raise ValueError("output_file_name must be provided if save_to_file is True")

    save_batch_results(output_file_id, output_file_name)


def save_batch_results(output_file_id, output_file_name):
    output_file = client.files.content(output_file_id)
    with open(output_file_name, "w") as f:
        f.write(output_file.text)
    print(f"Batch results saved to {output_file_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload and create a GPT-4o mini batch job."
    )
    parser.add_argument("mode", choices=["write", "send", "validate"])
    parser.add_argument(
        "--json_file",
        type=str,
        help="Path to the JSON file for writing batch files.",
    )
    parser.add_argument(
        "--batch_file",
        type=str,
        required=True,
        help="Path to the batch file to be sent.",
    )
    parser.add_argument(
        "--description", type=str, required=True, help="Description of the batch job."
    )
    parser.add_argument("--e2", action="store_true", help="Experiment 2")
    parser.add_argument(
        "--max_results",
        type=int,
        default=None,
        help="Maximum number of results to process from the batch file.",
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        default=False,
        help="Flag to indicate if the model output to validate human nouns from is local or proprietary.",
    )
    parser.add_argument("--dataset", type=str, help="Dataset name for the batch job.")
    parser.add_argument(
        "--content_type", type=str, help="Content type for the batch job."
    )
    parser.add_argument(
        "--output_file_id", type=str, help="ID of the output file to save results."
    )
    parser.add_argument(
        "--output_file_name", type=str, help="Name of the output file to save results."
    )
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        exit(1)

    if args.mode == "write":
        if not args.json_file or not args.dataset or not args.content_type:
            raise ValueError(
                "For write mode, --json_file, --dataset, and --content_type must be provided."
            )
        write_batch_file(
            args.json_file,
            args.dataset,
            args.content_type,
            is_e2=args.e2,
            is_local=args.is_local,
            max_results=args.max_results,
        )
    elif args.mode == "send":
        if not args.batch_file or not args.description:
            raise ValueError(
                "Both --batch_file and --description must be provided for send mode."
            )
        uploaded_batch_file = upload_batch(args.batch_file)
        send_batch(uploaded_batch_file, args.description)
    elif args.mode == "validate":
        if not args.output_file_id:
            raise ValueError("For validate mode, --output_file_id must be provided.")
        batch_results = get_batch_results(args.output_file_id, save_to_file=False)
        validate_output(batch_results)


if __name__ == "__main__":
    main()
