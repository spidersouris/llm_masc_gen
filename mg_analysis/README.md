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

`python filtering.py [-h] [--e2] {oasst2,oracle,hh_rlhf,alpaca}`

This requires spaCy and the `fr_dep_news_trf` and `fr_core_news_lg` spaCy models:

```
python -m spacy download fr_dep_news_trf
python -m spacy download fr_core_news_lg
```

Use `--e2` to get MG-only instructions from Experiment 2.

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

`python mscore.py [-h] [--results RESULTS [RESULTS ...]] [--results_mgonly RESULTS_MGONLY [RESULTS_MGONLY ...]] [--datasets DATASETS [DATASETS ...]] [--e2] [--mg_count_range MG_COUNT_RANGE MG_COUNT_RANGE] [--z_score Z_SCORE]`

Results and datasets are arrays. Should be in the same order. Example:

`python mscore.py --results instr_outputs_mg_results/real/llm_prop/claude-3-haiku_response_results_final.json instr_outputs_mg_results/real/llm_prop/gpt4o_mini_response_results_final.json --results_mgonly instr_outputs_mg_results/real/llm_prop/claude-3-haiku_response_results_mgonly_final.json instr_outputs_mg_results/real/llm_prop/gpt4o_mini_response_results_mgonly_final.json --datasets claude-3-haiku gpt4o_mini`

| Argument                                               | Description                                                                                                                            |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `--results RESULTS [RESULTS ...]`                      | Paths to JSON files containing results from **Experiment 1**.                                                                          |
| `--results_mgonly RESULTS_MGONLY [RESULTS_MGONLY ...]` | Paths to JSON files containing results from **Experiment 2 (MG only)**.                                                                |
| `--datasets DATASETS [DATASETS ...]`                   | Names of the datasets ordered by the order of the results files.                                                                       |
| `--e2`                                                 | Show **Experiment 2** plot. Applies to plots: _Classes_, _Z-Plot MG Count_. Other plots show both experiments’ results simultaneously. |
| `--mg_count_range MG_COUNT_RANGE MG_COUNT_RANGE`       | _(Z-Plot MG Count only)_ Range of MG counts to consider.                                                                               |
| `--z_score Z_SCORE`                                    | _(Z-Plot MG Count only)_ Z-Score threshold for outlier detection.                                                                      |

To plot, make sure to install `kaleido`.

`pip install --user kaleido`
