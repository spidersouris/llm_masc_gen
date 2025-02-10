import json
import os
from tqdm import tqdm
from get_llm_instrs import load_instructions


def infer_openrouter():
    from openai import OpenAI

    model = ""  # model name

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="",  # OR api key
    )

    processed_instr = []
    responses = []
    sources = []
    neutrals = []
    instructions = load_instructions(small_sample=True)

    failed_instructions = []
    failed_instructions_idx = []

    for i, (instruction, source, neutral) in enumerate(
        tqdm(
            zip(
                instructions["instruction"],
                instructions["source"],
                instructions["neutral"],
            ),
            total=len(instructions),
        )
    ):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful French assistant.",
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": instruction}],
                    },
                ],
                temperature=1,
                max_completion_tokens=1500,
                top_p=1,
                stream=False,
                stop=None,
                seed=42,
            )
            processed_instr.append(instruction)
            responses.append(response.choices[0].message.content)
            sources.append(source)
            neutrals.append(neutral)
        except Exception as e:
            print(f"\n\n!!!! Error with instruction: {instruction}")
            print(f"\nError: {e}")
            failed_instructions.append(instruction)
            failed_instructions_idx.append(i)

    with open(
        f"llm_responses/{model.split('/')[1]}_responses.json",
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

    print(f"\n\nResponses saved to llm_responses/{model}_responses.json")


def infer_openai_batch():
    def format_batch(i, prompt, temp):
        return {
            "custom_id": f"{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful French assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1500,
                "temperature": temp,
            },
        }

    print("Creating batch")

    instr = load_instructions(small_sample=True)
    sources = instr["source"].to_list()
    data = instr["instruction"].to_list()
    neutral_instr = instr["neutral"].to_list()

    requests = []

    for i, d, s, n in zip(range(len(data)), data, sources, neutral_instr):
        if i % 100 == 0:
            print(f"Processing item {i+1}/{len(data)}")

        requests.append(format_batch(f"{s}_{i}_{str(int(n))}", d, 1.0))

    if not os.path.exists("batch_instrs"):
        os.makedirs("batch_instrs", exist_ok=True)

    with open("batch_instrs/gpt_batch_instrs.jsonl", "w") as f:
        f.write("")

    for request in requests:
        with open("batch_instrs/gpt_batch_instrs.jsonl", "a") as f:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    client = OpenAI()

    def upload_batch(batch_file):
        batch_input_file = client.files.create(
            file=open(batch_file, "rb"), purpose="batch"
        )

        return batch_input_file

    def create_batch(uploaded_batch_file, description, completion_window="24h"):
        batch_input_file_id = uploaded_batch_file.id
        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
                "description": description,
            },
        )

        print(f"Batch {description} created successfully!")

    def get_batch_results(output_file_id, output_file_name=None, save_to_file=True):
        if not save_to_file:
            return client.files.content(output_file_id).text

        if output_file_name is None:
            raise ValueError(
                "output_file_name must be provided if save_to_file is True"
            )

        save_batch_results(output_file_id, output_file_name)

    def save_batch_results(output_file_id, output_file_name):
        output_file = client.files.content(output_file_id)
        with open(output_file_name, "w") as f:
            f.write(output_file.text)
        print(f"Batch results saved to {output_file_name}")


def infer_claude_batch():
    CLAUDE_API_KEY = ""  # Claude API key

    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    instr = load_instructions(small_sample=True)

    print("Creating batch")

    sources = instr["source"].to_list()
    data = instr["instruction"].to_list()
    neutral_instr = instr["neutral"].to_list()

    requests = []

    for i, d, s, n in zip(range(len(data)), data, sources, neutral_instr):
        if i % 100 == 0:
            print(f"Processing item {i+1}/{len(data)}")

        params = MessageCreateParamsNonStreaming(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            temperature=1,
            system="You are a helpful French assistant.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": d,
                        },
                    ],
                }
            ],
        )

        request = Request(custom_id=f"{s}_{i}_{str(int(n))}", params=params)

        requests.append(request)

        message_batch = client.messages.batches.create(requests=requests)
        print(message_batch)


def infer_gemini():
    import google.generativeai as genai

    GEMINI_KEY = ""  # Gemini API key

    genai.configure(api_key=GEMINI_KEY)

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction="You are a helpful French assistant.",
    )

    processed_instr = []
    responses = []
    sources = []
    neutrals = []
    instructions = load_instructions(small_sample=True)

    for instruction, source, neutral in tqdm(
        zip(
            instructions["instruction"], instructions["source"], instructions["neutral"]
        ),
        total=len(instructions),
    ):
        response = (
            model.generate_content(
                instruction,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=1500,
                    temperature=1,
                ),
            ),
        )
        processed_instr.append(instruction)
        if response[0]:
            responses.append(response[0].candidates[0].content.parts[0].text)
        else:
            print(response)
        sources.append(source)
        neutrals.append(neutral)

    with open("llm_responses/gemini_responses.json", "w", encoding="utf-8") as f:
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

    print("Responses saved to llm_responses/gemini_responses.json")
