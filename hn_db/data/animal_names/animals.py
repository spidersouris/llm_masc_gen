from bs4 import BeautifulSoup
import requests
import csv
import locale

locale.setlocale(locale.LC_ALL, "fr_FR.UTF-8")


def extract_animal_names(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    tables = soup.find_all("table", {"class": "wikitable"})

    animal_names = []
    for table in tables:
        rows = table.find_all("tr")[1:]
        for row in rows:
            first_cell = row.find("td")
            if first_cell:
                link = first_cell.find("a")
                if link:
                    animal_names.append(link.text.strip().split()[0])
                else:
                    animal_names.append(first_cell.text.strip().split()[0])

    animal_names.remove("humain")

    animal_names = sorted(
        set(animal_names[: animal_names.index("aboiement")]), key=locale.strxfrm
    )

    return animal_names


url = "https://fr.wiktionary.org/wiki/Annexe:Animaux_communs_en_fran%C3%A7ais"
response = requests.get(url)
html_content = response.text
animal_names = extract_animal_names(html_content)

with open("animals.csv", "w", encoding="utf8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["animal"])
    for animal in animal_names:
        writer.writerow([animal])
