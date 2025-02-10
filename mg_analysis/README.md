## Create Masculine Generics Dictionary

See [llm_masc_gen_dict.ipynb](llm_masc_gen_dict.ipynb). Resulting DF is in [dfs/masc_gen_df.pkl](dfs/masc_gen_df.pkl).

## Create Instruction Datasets

`python create_instr_dataset.py [-h] {oasst2,oracle,hh_rlhf,alpaca}`

This requires spaCy and the `fr_dep_news_trf` spaCy model:

```
python -m spacy download fr_dep_news_trf
```

DFs will be saved to the corresponding folders in `dfs/{dataset}` as `{dataset}_df.pkl`.

## Filter Specific Instructions

`python filtering.py [-h] {oasst2,oracle,hh_rlhf,alpaca}`

This requires spaCy and the `fr_dep_news_trf` and `fr_core_news_lg` spaCy models:

```
python -m spacy download fr_dep_news_trf
python -m spacy download fr_core_news_lg
```

DFs will be saved to the corresponding folders in `dfs/{dataset}` as `{dataset}_filtered_df.pkl`.

## Send Instructions to LLMs

If you want to use local models on your machine, see [infer_llm_local.py](infer_llm_local.py).

If you want to use proprietary models or local models from the OpenRouter API, see [infer_llm.py](infer_llm.py).

## GPT-4o mini Human Noun Validation

`python gpt_eval.py [-h] [--gpt_output_files GPT_OUTPUT_FILES [GPT_OUTPUT_FILES ...]]
                   [--original_results_files ORIGINAL_RESULTS_FILES [ORIGINAL_RESULTS_FILES ...]]
                   [--final_files FINAL_FILES [FINAL_FILES ...]] [--positive_only]`

## Masculine Generics Analysis

Used to create the JSON MG analysis file to be used to compute scores.

This requires spaCy and the `fr_dep_news_trf` and `fr_core_news_lg` spaCy models:

```
python -m spacy download fr_dep_news_trf
python -m spacy download fr_core_news_lg
```

Will create output file as `instr_outputs_mg_results/{real_folder}/{dataset_group}/{dataset}_{content_type}_results.json`.

`python mg_analysis.py [-h] [--content_type CONTENT_TYPE] [--dataset DATASET] [--is_real] df_path`

## Compute Scores and Visualization

`python mscore.py [-h] [--results RESULTS [RESULTS ...]] [--datasets DATASETS [DATASETS ...]]`

Results and datasets are arrays. Should be in the same order. Example:

`python mscore.py --results instr_outputs_mg_results/real/llm_prop/claude-3-haiku_response_results_final.json instr_outputs_mg_results/real/llm_prop/gpt4o_mini_response_results_final.json --datasets claude-3-haiku gpt4o_mini`

To plot, make sure to install `kaleido`.

> [!WARNING]
> If you're on Windows 11, install `kaleido==0.1.0post1`. Installing another version and launching using the command line will cause a hang on Windows 11. See [this plotly issue](https://github.com/plotly/Kaleido/issues/126) for more information.

`pip install --user kaleido`