usage: `annot.py [-h] [--create-tsv json_file output_tsv] [--kappa annotator1_tsv annotator2_tsv] [--kappa_gpt agreed_tsv]`

Script to handle annotations.

options:
```bash
  -h, --help            show this help message and exit
  --create-tsv json_file output_tsv
                        Create TSV file for annotation.
  --kappa annotator1_tsv annotator2_tsv
                        Calculate Kappa score between two annotators.
  --kappa_gpt agreed_tsv
                        Calculate Kappa score between GPT-4o mini and agreed annotation.
```