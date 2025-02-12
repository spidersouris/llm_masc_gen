This folder contains a custom-made script used to scrape TLFi data on https://www.cnrtl.fr/definition/.

The script was made using TypeScript and the [playwright](https://playwright.dev/) package.

Usage: tlfi_scraper [options]

Scrape the TLF website for word definitions and other linguistic features.

Options:

```
  -V, --version       output the version number
  -w, --words <file>  File containing the words to scrape. Can be JSON or CSV (see examples in /words folder). If no word file is provided, a predefined 'human noun search' will first be made in order to retrieve relevant words to scrape and add them to a JSON file.
  -r, --recursive     Recursive search
  -h, --help          Show help
```

  ## Databases

  The /dbs folder contains three different databases:
  - db_tlfi_complete.db: database of all TLFi noun definition and linguistic features extracted from words in words/words.json (recursive excluded)
  - db/tlfi_recursive.db: database of TLFi noun definition and linguistic features extracted by doing recursive search. Recursive search works like this: we first create a new words.json file by using nouns in db_tlfi_complete.db as the first word in the definition (for instance, instead of searching "&d0 personne", we search "&d0 auteur"). Then, when the word file is generated, we use that word file to get definition and linguistic features and we populate the tlfi_recursive.db database.
  - db/tlfi_rec_pos.db: database of recursive nouns that were considered ground-truth by the 3 models (LR, XGBoost, Transformer).