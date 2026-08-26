# Masculine Generics Analysis

This folder contains the full pipeline used to measure masculine generics (MG) use in
human-written and LLM-generated French texts.

Two experiments were conducted:

- **Experiment 1 (E1)**: LLMs are prompted with instructions sampled from four corpora
  (`oasst2`, `oracle`, `hh_rlhf`, `alpaca`) + comparison with human-written datasets `oracle` and `oasst2`
- **Experiment 2 (E2)**: LLMs are prompted only with instructions that contain MG.

If you only want to regenerate the figures and tables of the paper from the results
files already committed in this repository, check
[Reproducing the paper's figures and tables](#reproducing-the-papers-figures-and-tables).

___

## Requirements

Install the repository requirements first (from the repository root):

```
pip install -r requirements.txt
```

### spaCy models

`create_instr_dataset.py`, `filtering.py`, `get_llm_instrs.py` and `mg_analysis.py`
require two French spaCy models:

```
python -m spacy download fr_dep_news_trf
python -m spacy download fr_core_news_lg
```

### Plotting

Figure rendering uses Plotly and Kaleido (which needs to be installed first):

```
pip install --user kaleido
```

The `--pdf` flag additionally converts each SVG to PDF using Typst's
[`svg2pdf`](https://github.com/typst/svg2pdf) CLI, which must be on your `PATH`:

```
cargo install svg2pdf-cli
```

`svg2pdf` is used as an alternative to Kaleido's built-in PDF export because it produces
noticeably better vector output. Without `--pdf`, only SVGs are written and `svg2pdf`
is not needed.

___

## Steps


### 1. Create Masculine Generics Dictionary

See [llm_masc_gen_dict.ipynb](llm_masc_gen_dict.ipynb). Resulting DF is in
[dfs/masc_gen_df.pkl](dfs/masc_gen_df.pkl).

### 2. Create Instruction Datasets

```
python create_instr_dataset.py [-h] {oasst2,oracle,hh_rlhf,alpaca}
```

This requires spaCy and the `fr_dep_news_trf` spaCy model.

DFs will be saved to the corresponding folders in `dfs/{dataset}` as `{dataset}_df.pkl`.

### 3. Filter Specific Instructions

```
python filtering.py [-h] [--e2] {oasst2,oracle,hh_rlhf,alpaca}
```

| Argument | Description                                                                |
| -------- | -------------------------------------------------------------------------- |
| `--e2`   | Keep only instructions containing masculine generics (**Experiment 2**).   |

This requires spaCy and the `fr_dep_news_trf` and `fr_core_news_lg` spaCy models.

DFs will be saved to the corresponding folders in `dfs/{dataset}` as `{dataset}_filtered_df.pkl`

### 4. Get Instructions for Inference

Use [get_llm_instrs.py](get_llm_instrs.py) to get a
single instruction DF to be used for LLM inference.

```
python get_llm_instrs.py [-h] [--e2]
```

| Argument | Description                                                              |
| -------- | ------------------------------------------------------------------------ |
| `--e2`   | Pool MG only instructions from **Experiment 2**.                         |

DF will be saved to `dfs/instructions_llm_inference.pkl`, or to
`dfs/instructions_mg_only.pkl` with `--e2`.

### 5. Send Instructions to LLMs

If you want to use local models on your machine, see [infer_llm_local.py](infer_llm_local.py).

If you want to use proprietary models or local models from the OpenRouter API, see [infer_llm.py](infer_llm.py).

### 6. Masculine Generics Detection

Creates the JSON MG analysis file later used to compute scores.

```
python mg_analysis.py [-h] [--content_type CONTENT_TYPE] [--dataset DATASET] [--is_real] df_path
```

| Argument                        | Default       | Description                                                       |
| ------------------------------- | ------------- | ----------------------------------------------------------------- |
| `df_path`                       |   | DF to analyze (`*_filtered_df.pkl`)           |
| `--content_type CONTENT_TYPE`   | `instruction` | Column substring to analyze |
| `--dataset DATASET`             | `oasst2`      | Dataset name                    |
| `--is_real`                     | `False`       | Mark results as GPT-validated (writes under `real/` instead of `unreal/`) |

Output: `instr_outputs_mg_results/{real|unreal}/{dataset_group}/{dataset}_{content_type}_results.json`.

### 7. GPT-4o mini Human Noun Validation

Each candidate human noun detected in step 6 is validated in context by GPT-4o mini.

The exact system and user prompts are in [PROMPTS.md](../PROMPTS.md).

```
python gpt_eval.py [-h] [--gpt_output_files GPT_OUTPUT_FILES [GPT_OUTPUT_FILES ...]]
                   [--original_results_files ORIGINAL_RESULTS_FILES [ORIGINAL_RESULTS_FILES ...]]
                   [--final_files FINAL_FILES [FINAL_FILES ...]] [--positive_only]
```

| Argument                   | Description                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `--gpt_output_files`       | GPT batch output files to process (`.jsonl`)                                               |
| `--original_results_files` | Results files from step 6 to annotate (`.json`)                                            |
| `--final_files`            | Destination paths for the validated results (`.json`)                                      |
| `--positive_only`          | Keep only tokens validated as human nouns. Use for analysis and omit for eval-data generation. |

All files arguments are arrays. Must be speciifed in the same order and must be the same length.

The resulting `*_results_final.json` files are the inputs to `mscore.py`.

Inter-annotator agreement for the validation study is computed with
[eval/annot/annot.py](eval/annot/README.md).

___

## Reproducing the paper's figures and tables

When `--results`, `--results_mgonly` and `--datasets` are omitted, `mscore.py` uses the
default set of eight datasets (`oracle`, `oasst2`, `gemini`, `gpt4o_mini`,
`claude-3-haiku`, `llama`, `ministral`, `mistral-small`) and resolves their result files
under `instr_outputs_mg_results/real/`.

In case different results and datasets are passed as arguments, they should be in the same order. Example:

```
python mscore.py --results instr_outputs_mg_results/real/llm_prop/claude-3-haiku_response_results_final.json instr_outputs_mg_results/real/llm_prop/gpt4o_mini_response_results_final.json --results_mgonly instr_outputs_mg_results/real/llm_prop/claude-3-haiku_response_results_mgonly_final.json instr_outputs_mg_results/real/llm_prop/gpt4o_mini_response_results_mgonly_final.json --datasets claude-3-haiku gpt4o_mini
```

Make sure to run the confidence intervals before creating the plots.

### Confidence intervals (bootstrap, `n = 10 000`, seed 42)

```
python mscore.py --ci br       # creates analyses/bias_rates.csv
python mscore.py --ci markers  # creates analyses/marker_rates.csv
```

### Main plots and Pareto analysis

```
python mscore.py --pdf       # Experiment 1
python mscore.py --e2 --pdf  # Experiment 2
```

`--e2` only switches the _Classes_ and _MG count_ plots between experiments. The bias
rate and language marker plots always show both experiments side by side.

### Epicene-excluded MG counts

```
python mscore.py --filter-epicenes      # creates analyses/mg_count_detailed_e1_no_epicenes.csv
python mscore.py --e2 --filter-epicenes # creates analyses/mg_count_detailed_e2_no_epicenes.csv
```

`--filter-epicenes` drops MG occurrences whose lemma is epicene
from the MG count analysis only. It does **not** affect bias rates, marker rates or the
Pareto analysis.

### Pareto front stability

```
python mscore.py --pareto_only --ci pareto --n_boot_pareto 10000
```

Adds a `P(Pareto)` column to `analyses/pareto.csv`, giving the bootstrap probability
that each model is on the Pareto front.

___

## `mscore.py` reference

```
python mscore.py [-h] [--results RESULTS [RESULTS ...]]
                 [--results_mgonly RESULTS_MGONLY [RESULTS_MGONLY ...]]
                 [--datasets DATASETS [DATASETS ...]] [--e2]
                 [--mg_count_range MG_COUNT_RANGE MG_COUNT_RANGE] [--z_score Z_SCORE]
                 [--filter-epicenes] [--weights W W W W W] [--ci {br,pareto,markers}]
                 [--pdf] [--legacy] [--n_boot_pareto N_BOOT_PARETO]
                 [--log_level {DEBUG,INFO,WARNING,ERROR}] [--pareto_only]
```

| Argument                                               | Default   | Description                                                                                                                      |
| ------------------------------------------------------ | --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `--results RESULTS [RESULTS ...]`                      | all  | Paths to JSON files containing results from **Experiment 1**.                                                                    |
| `--results_mgonly RESULTS_MGONLY [RESULTS_MGONLY ...]` | all  | Paths to JSON files containing results from **Experiment 2 (MG only)**.                                                          |
| `--datasets DATASETS [DATASETS ...]`                   | all  | Names of the datasets ordered by the order of the results files.                                                                           |
| `--e2`                                                 | `False`   | Show **Experiment 2** plots. Applies to plots: _Classes_, _Z-Plot MG Count_. Other plots show both experiments’ results simultaneously.  |
| `--mg_count_range MIN MAX`                             | `0 50`    | _(Z-Plot MG Count only)_ Rank range of MG counts to display. |
| `--z_score Z_SCORE`                                    | `1.7`     | _(Z-Plot MG Count only)_ Z-score threshold for outlier detection.                                                               |
| `--filter-epicenes`                                    | `False`   | _(MG count only)_ Exclude MG occurrences derived from epicene nouns.                |
| `--ci {br,pareto,markers}`                             |   | `br` / `markers`: bootstrap bias rates/markers CIs. `pareto`: bootstrap Pareto-front membership (combine with `--pareto_only`). |
| `--n_boot_pareto N`                                    | `10000`   | Bootstrap iterations for the Pareto stability analysis.                                                                          |
| `--pareto_only`                                        | `False`   | Run the Pareto analysis only (skips all other plots).                                                                 |
| `--pdf`                                                | `False`   | Also write PDF versions of the plots to `plots/pdfs/` (requires `svg2pdf` on `PATH`).                                            |
| `--log_level {DEBUG,INFO,WARNING,ERROR}`               | `DEBUG`   | Console logging verbosity.                                                                                                       |
| `--legacy`                                             | `False`   | _(shelved)_ Use the legacy MScore formula (MG/HN ratio only).                          |
| `--weights W W W W W`                                  |   | _(shelved)_ Custom weights for `[mg, inclusive, inclusive_greetings, neutral, neutral_pronouns]` in the MScore formula.          |
