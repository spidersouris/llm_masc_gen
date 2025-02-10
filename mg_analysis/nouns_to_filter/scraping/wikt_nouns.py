import requests
from bs4 import BeautifulSoup
import re


def scrape_wiktionary_nouns(url, selected_sections=None, excluded_words=None):
    selected_sections = selected_sections or []
    excluded_words = set(excluded_words or [])

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    main_div = soup.find("div", class_="mw-content-ltr mw-parser-output")
    if not main_div:
        print("Could not find main content div")
        return []

    all_nouns = []

    sections = main_div.find_all("h2")
    for section in sections:
        section_title = section.get_text(strip=True)

        if selected_sections and section_title not in selected_sections:
            continue

        current_element = section.find_next()
        print(current_element)
        while current_element and current_element.name != "h2":
            if current_element.name == "p":
                bold_element = current_element.find(
                    "b", string=re.compile(r"Noms", re.IGNORECASE)
                )
                if bold_element:
                    noun_links = current_element.find_all("a")
                    for link in noun_links:
                        noun = link.get_text(strip=True)
                        if noun and noun not in excluded_words:
                            all_nouns.append(noun)
            current_element = current_element.find_next()

    unique_nouns = list(dict.fromkeys(all_nouns))

    return unique_nouns


def save_nouns_to_file(nouns, filename="scraped_nouns.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for noun in nouns:
                f.write(noun + "\n")
        print(f"Nouns saved to {filename}")
    except IOError as e:
        print(f"Error saving nouns to file: {e}")


def main():
    url = "https://fr.wiktionary.org/wiki/Wiktionnaire:Liste_de_1750_mots_fran%C3%A7ais_les_plus_courants"
    excluded_words = [
        "ami",
        "camarade",
        "copain",
        "dame",
        "directeur",
        "directrice",
        "élève",
        "enfant",
        "fille",
        "garçon",
        "gardien",
        "madame",
        "monsieur",
        "maîtresse",
        "personne",
        "magicien",
        "soldat",
        "sorcière",
        "bébé",
        "bébés",
        "gens",
        "grand-mère",
        "grand-père",
        "géant",
        "bonhomme",
        "princesse",
        "adulte",
        "mère",
        "papa",
        "parent",
        "maman",
        "père",
        "petite-fille",
        "petit-enfant",
        "petit-fils",
        "sœur",
        "mari",
        "jumeau",
        "homme",
        "femme",
        "dentiste",
        "docteur",
        "infirmière",
        "médecin",
        "chasseur",
        "clown",
        "coiffeur",
        "facteur",
        "fleuriste",
        "pharmacien",
        "pompier",
        "policier",
        "voisin",
        "pêcheur",
        "marin",
        "coquin",
    ]

    nouns = scrape_wiktionary_nouns(
        url,
        # selected_sections=selected_sections,
        excluded_words=excluded_words,
    )

    print("Scraped Nouns:")
    print(nouns)

    print("Length:")
    print(len(nouns))

    save_nouns_to_file(nouns)


if __name__ == "__main__":
    main()
