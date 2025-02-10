## Get golden human/non-human dataset

`python data/create_datasets.py [-h] [--wikidata_db_path WIKIDATA_DB_PATH] [--wiktionary_db_path WIKTIONARY_DB_PATH] [--nhuma_csv_path NHUMA_CSV_PATH] [--label LABEL]`

If paths are not specified, default paths will be used.

If label is not specified, "int" will be used.

## Train

`python train.py [-h] human_pkl_path non_human_pkl_path {lr,xgboost,transformer} output_file`

Training requires having `cc.fr.300.bin ` in the `data/ft/` folder (you should create it). French FastText 300-dimension word vectors can be downloaded from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz).

## Classify

`python classify.py [-h] [--disagreement_type DISAGREEMENT_TYPE] {tlfi_recursive,demonette} sum_prob positive_df_path disagreement_df_path`

Classifying requires having `cc.fr.300.bin ` in the `data/ft/` folder (you should create it). French FastText 300-dimension word vectors can be downloaded from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz).