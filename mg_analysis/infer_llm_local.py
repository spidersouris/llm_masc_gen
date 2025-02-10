import json
import torch
from tqdm import tqdm
from get_llm_instrs import load_instructions
from codecarbon import EmissionsTracker
from transformers import pipeline


def infer_local():
    def get_llm_response(model_name, instruction):
        prompt = [
            {"role": "system", "content": "You are a helpful French assistant."},
            {"role": "user", "content": instruction},
        ]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        generator = pipeline(
            model=model_name, device=device, torch_dtype=torch.bfloat16
        )
        generation = generator(
            prompt, do_sample=False, temperature=1.0, top_p=1, max_new_tokens=1500
        )

        return generation[0]["generated_text"]

    MODELS = []  # define HF models here

    instructions = load_instructions(small_sample=True)

    for model in MODELS:
        with EmissionsTracker(
            save_to_file=True,
            save_to_api=False,
            measure_power_secs=30,
            experiment_id=f"llm_inference_{model}",
        ) as tracker:
            processed_instr = []
            responses = []
            sources = []
            neutrals = []
            instructions = load_instructions()

            print(f"Model: {model}")

            for instruction, source, neutral in tqdm(
                zip(
                    instructions["instruction"],
                    instructions["source"],
                    instructions["neutral"],
                ),
                total=len(instructions),
            ):
                response = get_llm_response(model, instruction)
                processed_instr.append(instruction)
                responses.append(response)
                sources.append(source)
                neutrals.append(neutral)

            with open(
                f"llm_responses/{model}_responses.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "instruction": processed_instr,
                        "response": responses,
                        "source": sources,
                        "neutral": neutrals,
                    },
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

            print(f"Responses saved to llm_responses/{model}_responses.json")
