import requests
from bs4 import BeautifulSoup
import csv

url = "https://en.wiktionary.org/wiki/Appendix:French_demonyms"
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

tables = soup.find_all("table")

demonyms = []

for table in tables:
    rows = table.find_all("tr")
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) >= 3:
            demonym = cells[2].get_text(strip=True)
            demonym = demonym.split("(")[0].strip()

            for demonym in demonym.split(","):
                demonyms.append(demonym.strip())

with open("demonyms.csv", mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Demonym"])
    for demonym in demonyms:
        writer.writerow([demonym])

print("Data has been successfully scraped and saved to 'demonyms.csv'.")
