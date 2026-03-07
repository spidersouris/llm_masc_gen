# System Prompt

You are an assistant that validates human noun classifications in French texts.

# User Prompt

Given a text and nouns, for each noun, determine if it is a human noun in context. Some nouns may appear multiple times in the text. In such cases, they are distinguished by ID ('noun_1', 'noun_2'...), following the order in which they appear. Do not assume that all occurrences of the same noun are either human or non-human; instead, assess each occurrence individually based on its unique context. Only respond in this format, where human_noun is the noun being considered.
{
  "human_noun": 0,
  "human_noun_2": 1
}

## Examples
Text: Les facteurs d'employabilité des facteurs, chargés de distribuer le courrier, vont évoluer.
Nouns: facteurs, facteurs_2
Output: { "facteurs": 0, "facteurs_2": 1 }

Text: Le président a annoncé aux citoyens une série de mesures pour renforcer l'économie du pays.
Nouns: président, citoyens, mesures
Output: { "président": 1, "citoyens": 1, "mesures": 0 }

Text: Il croit aux esprits et aux fantômes depuis qu'il est enfant.
Nouns: esprits, fantômes, enfant
Output: { "esprits": 0, "fantômes": 0, "enfant": 1 }

Text: {text}
Nouns: {human_nouns}
Output:
