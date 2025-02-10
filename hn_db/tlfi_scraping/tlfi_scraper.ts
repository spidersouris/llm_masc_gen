import { Database } from "./tlfi_db";
import {
  WordEl,
  WordDefinitionFeatures,
  DefinitionExample,
  SectionKey,
  Sections,
} from "./types";

import { Page } from "playwright";
import { chromium } from "playwright-extra";
import StealthPlugin from "puppeteer-extra-plugin-stealth";

import { stripHtml } from "string-strip-html";

import csv from "convert-csv-to-json";

import { Command } from "commander";

import fs from "fs";

import * as rl from "readline";

const rlInterface = rl.createInterface({
  input: process.stdin,
  output: process.stdout,
});

chromium.use(StealthPlugin());

const db = new Database();

const excludedWords = ["alopécique", "anonymographe"];

// These words cause problem with the default splitting logic (space)
// in getWordsOnPage()
const specialSplitWords = ["vice-premier", "alter ego", "persona non grata"];
// These words are not formatted correctly on the website
// E.g. "cryptocommuniste" is formatted as "cryptocommunistesubst."
// See https://www.cnrtl.fr/definition/cryptocommuniste
const badlyFormattedWords = ["cryptocommuniste"];

const genderSuffixes = {
  eur: "euse",
  ard: "arde",
  er: "ère",
  us: "use",
};

// function nodeSearch(
//   el: HTMLElement | SVGElement,
//   className: string,
//   maxDepth: number = 3
// ): string | null {
//   let currentElement: HTMLElement | SVGElement | null = el;
//   let depth = 0;

//   console.log("Searching for class:", className, "in element:", el);

//   while (currentElement && depth < maxDepth) {
//     // Check if the current element has the target class
//     if (currentElement.classList.contains(className)) {
//       return currentElement.textContent;
//     }

//     // Check preceding siblings
//     let sibling = currentElement.previousElementSibling;
//     while (sibling) {
//       if (sibling.classList.contains(className)) {
//         return sibling.textContent;
//       }
//       sibling = sibling.previousElementSibling;
//     }

//     // Move up to the parent
//     currentElement = currentElement.parentElement;
//     depth++;
//   }

//   return null;
// }

function cleanContent(content: string, cleanPunct: boolean = false): string {
  console.log(`Content before: ${content}`);
  const stripped = stripHtml(content).result.trim();

  if (cleanPunct) {
    // need to trim a second time after stripping HTML
    return stripped.replace(/(^(, )|,$)/g, "").trim();
  }

  return stripped;
}

async function submitWordList(page: Page): Promise<void> {
  const textArea = page.locator("textarea[name='texte']");

  // fill the textarea with the word list
  //! too many false positives with "celui" and "celle"
  //! and since the word list only allows one word per line,
  //! we cannot add "celui, celle" or "celui ou celle"
  //! do the search separately and add to words.json?
  await textArea.fill("personne\nindividu\nquelqu'un\nhomme\nfemme");

  const inputListName = page.locator("input[name='liste']").first();

  // fill the name of the list
  await inputListName.fill("nhumains");

  // click input button
  await page.locator("input[type='button'][value='Valider 1']").click();

  console.log("Word list created");
}

//! IMPORTANT: deal with hierarchy issue
//! see hierarchy_issue.png in folder for more info
async function scrapeWordPage(
  page: Page,
  wordEls: WordEl[]
): Promise<WordEl[] | null> {
  const highlightEls = {
    wordDefinition: "tlf_cdefinition",
    wordExamples: "tlf_cexemple",
    wordIndicator: "tlf_cemploi",
    wordBracketText: "tlf_ccrochet",
  };

  const words: WordEl[] = [];

  // this function should only be ran if there are other definitions
  // (check if count of //span[@class="stab0"] is greater than 1)
  // if true, pass them as parameters.
  //
  // We should perform a check in another function to see if these
  // definitions contain "personne", and the other words of our list
  //
  // For each definition, we need to get:
  // - wordDefinition
  // - wordExamples
  // - wordIndicator
  // - wordPhrase
  // - wordBracketText
  // so the current interfaces should be changed to reflect this
  for (const wordEl of wordEls) {
    console.log(`Scraping word: ${wordEl.wordForm}...`);

    // handle cases where url is not provided in the words file
    // (e.g. when getting TLFi information from HScore-filtered recursive words)
    if (!wordEl.url) {
      wordEl.url = "https://www.cnrtl.fr/definition/" + wordEl.wordForm;
    }

    await page.goto(wordEl.url, { waitUntil: "networkidle" });

    const noResultsEl = await page
      .locator("div#contentbox h1", {
        hasText: "Cette forme est introuvable !",
      })
      .isVisible();

    if (noResultsEl) {
      console.error(
        "CRITICAL ERROR: No results found for word:",
        wordEl.wordForm
      );
      console.error(
        `This will be logged to errors/${wordEl.wordForm}_error.log`
      );
      fs.writeFileSync(
        `errors/${wordEl.wordForm}_error.log`,
        `word: ${wordEl.wordForm}\ncomplete word form: ${wordEl.completeWordForm}\nurl: ${wordEl.url}`
      );
      continue;
    }

    const wordDefinitionFeatures: WordDefinitionFeatures[] = [];

    const wordsToHighlight = [
      "personne",
      "individu",
      "homme",
      "femme",
      "celui",
      "celle",
    ];

    const tabsCount = await page
      .locator(`//*[@id='vitemselected']/following-sibling::*`)
      .count();

    console.log(`Found ${tabsCount} tabs.`);

    //check the text content of each tab
    for (let i = 1; i <= tabsCount; i++) {
      const tabText = await page
        .locator(`//*[@id='vitemselected']/following-sibling::*[${i}]`)
        .textContent();
      console.log(`Tab ${i} text: ${tabText}`);

      if (tabText === wordEl.completeWordForm) {
        console.log("Found complete word form tab.");
        await page
          .locator(`//*[@id='vitemselected']/following-sibling::*[${i}]`)
          .click();

        await page.waitForTimeout(2000);

        break;
      }
    }

    const completeWordForm = await page
      .locator("//*[@id='vitemselected']")
      .textContent();

    const wordType = await page
      .locator("//*[@id='vitemselected']")
      .evaluate((el, word) => {
        // if the words file does not have completeWordForm,
        // we need to get the whole thing
        if (!word.completeWordForm) {
          // Remove the span element
          const spanElement = el.querySelector("span");
          if (spanElement) {
            spanElement.remove();
          }
        }
        return (el as HTMLElement).innerText
          .replace(new RegExp(/(\s?,\s$|,$)/, "gm"), "")
          .trim();
      }, wordEl);

    const nbDefinitions = await page
      .locator(`//span[@class='${highlightEls.wordDefinition}']`)
      .count();

    console.log(`Found ${nbDefinitions} definitions.`);

    const definitionTexts: string[] = [];
    for (let i = 0; i < nbDefinitions; i++) {
      const definitionLocator = page
        .locator(`//span[@class='${highlightEls.wordDefinition}']`)
        .nth(i);
      const definitionText = (await definitionLocator
        .textContent()
        .then((text) => {
          // remove trailing space and colon in definitions
          return text?.replace(/\s:$/, "").trim();
        })) as string;

      // seems to be necessary
      // e.g. "accompagnateur" has the same definition twice
      if (!definitionTexts.includes(definitionText)) {
        definitionTexts.push(definitionText);
      } else {
        console.log("Definition already exists:", definitionText);
        continue;
      }

      const bracketText = await definitionLocator.evaluate(async (el) => {
        let currentElement: HTMLElement | SVGElement | null = el;
        let depth = 0;
        let maxDepth = 3;

        while (currentElement && depth < maxDepth) {
          // check if the current element has the target class
          if (currentElement.classList.contains("tlf_ccrochet")) {
            return currentElement.textContent;
          }

          // check preceding siblings
          let sibling = currentElement.previousElementSibling;
          while (sibling) {
            if (sibling.classList.contains("tlf_ccrochet")) {
              return sibling.textContent;
            }
            sibling = sibling.previousElementSibling;
          }

          // move up to the parent
          currentElement = currentElement.parentElement;
          depth++;
        }

        return null;
      });

      console.log("Has tlf_ccrochet:", bracketText);

      const bracketRegex =
        /(pers\.|personne|humain|quelqu'un|qqun|homme|femme)/i;
      const isInterestingBracket = bracketText
        ?.toLowerCase()
        .match(bracketRegex);

      console.log("Is interesting bracket:", isInterestingBracket);

      console.log("Definition text:", definitionText);

      if (!definitionText) {
        console.error(
          "Could not find definition text for word:",
          wordEl.wordForm
        );
        return null;
      }

      // check if the definition contains any of the words to highlight
      for (const word of wordsToHighlight) {
        const regex = new RegExp(`((tout|toute)\\s)?\\b${word}\\b`, "gi");
        if (definitionText.toLowerCase().match(word)) {
          console.log(`Found interesting definition: ${word}`);
          // replace with a font tag to get the highlighted text more easily
          const newText = definitionText.replace(regex, (match) => {
            return `<font color="red">${match}</font>`;
          });
          await definitionLocator.evaluate((element, newText) => {
            element.innerHTML = newText;
          }, newText);

          console.log("Highlighted definition:", newText);
        }
      }
    }

    const highlightedWordsEls = await page
      .locator("//font[@color='red']")
      .all();

    console.log(`Found ${highlightedWordsEls.length} interesting definitions.`);

    // for every element, find definition next to it
    for (const highlightedWordEl of highlightedWordsEls) {
      const definitionLocator = highlightedWordEl.locator("..");
      const definitionText = await definitionLocator.textContent();

      const bracketText = await definitionLocator.evaluate(async (el) => {
        //! this is absolutely horrible repeating this
        //! but I don't know how to do it better :(:(:(
        let currentElement: HTMLElement | SVGElement | null = el;
        let depth = 0;
        let maxDepth = 3;

        while (currentElement && depth < maxDepth) {
          // check if the current element has the target class
          if (currentElement.classList.contains("tlf_ccrochet")) {
            return currentElement.textContent;
          }

          // check preceding siblings
          let sibling = currentElement.previousElementSibling;
          while (sibling) {
            if (sibling.classList.contains("tlf_ccrochet")) {
              return sibling.textContent;
            }
            sibling = sibling.previousElementSibling;
          }

          // move up to the parent
          currentElement = currentElement.parentElement;
          depth++;
        }

        return null;
      });
      const indicatorText = await definitionLocator.evaluate(async (el) => {
        let currentElement: HTMLElement | SVGElement | null = el;
        let depth = 0;
        let maxDepth = 3;

        while (currentElement && depth < maxDepth) {
          // check if the current element has the target class
          if (currentElement.classList.contains("tlf_cemploi")) {
            return currentElement.textContent;
          }

          // check preceding siblings
          let sibling = currentElement.previousElementSibling;
          while (sibling) {
            if (sibling.classList.contains("tlf_cemploi")) {
              return sibling.textContent;
            }
            sibling = sibling.previousElementSibling;
          }

          // move up to the parent
          currentElement = currentElement.parentElement;
          depth++;
        }

        return null;
      });

      const domainText = await definitionLocator.evaluate(async (el) => {
        let currentElement: HTMLElement | SVGElement | null = el;
        let depth = 0;
        let maxDepth = 3;

        while (currentElement && depth < maxDepth) {
          // check if the current element has the target class
          if (currentElement.classList.contains("tlf_cdomaine")) {
            return currentElement.textContent;
          }

          // check preceding siblings
          let sibling = currentElement.previousElementSibling;
          while (sibling) {
            if (sibling.classList.contains("tlf_cdomaine")) {
              return sibling.textContent;
            }
            sibling = sibling.previousElementSibling;
          }

          // move up to the parent
          currentElement = currentElement.parentElement;
          depth++;
        }

        return null;
      });

      const phraseText = await definitionLocator.evaluate(async (el) => {
        let currentElement: HTMLElement | SVGElement | null = el;
        let depth = 0;
        let maxDepth = 3;

        while (currentElement && depth < maxDepth) {
          // check if the current element has the target class
          if (currentElement.classList.contains("tlf_csyntagme")) {
            return currentElement.textContent;
          }

          // check preceding siblings
          let sibling = currentElement.previousElementSibling;
          while (sibling) {
            if (sibling.classList.contains("tlf_csyntagme")) {
              return sibling.textContent;
            }
            sibling = sibling.previousElementSibling;
          }

          // move up to the parent
          currentElement = currentElement.parentElement;
          depth++;
        }

        return null;
      });

      //console.log(bracketText, indicatorText, domainText, phraseText);

      const definitionExamples: DefinitionExample[] = [];

      try {
        const examples = await definitionLocator
          .locator("..//span[@class='tlf_cexemple']")
          .all();

        for (const example of examples) {
          const exampleTextContent = await example.innerHTML();

          const exampleText = cleanContent(
            exampleTextContent
              .split('<span class="tlf_cauteur">')[0]
              // remove the number at the beginning of the example
              .replace(/\d+\. /, "")
          );

          const authorMatch = exampleTextContent.match(
            /<span class="tlf_cauteur">(.*?)<\/span>/
          );
          const author = authorMatch ? cleanContent(authorMatch[1]) : "";

          const titleMatch = exampleTextContent.match(
            /<span class="tlf_ctitre">(.*?)<\/span>/
          );
          const title = titleMatch
            ? // fix strings with "tome" label
              cleanContent(titleMatch[1], true).replace(/,(t. \d+)/g, ", $1")
            : "";

          const dateMatch = exampleTextContent.match(
            /<span class="tlf_cdate">(.*?)<\/span>/
          );
          const date = dateMatch ? dateMatch[1].replace(",", "").trim() : "";

          definitionExamples.push({
            example: exampleText,
            metadata: {
              author,
              title,
              date,
            },
          });
        }
      } catch (error) {
        console.error(
          "Could not find examples for word:",
          wordEl.wordForm,
          error
        );
      }

      wordDefinitionFeatures.push({
        definition: definitionText || "",
        bracketText: bracketText || "",
        indicator: indicatorText || "",
        domain: domainText || "",
        phrase: phraseText || "",
        examples: definitionExamples,
      });
    }

    const parsedSections = [];
    const parothers = await page.locator("div.tlf_parothers").all();

    for (const parother of parothers) {
      const text = await parother.innerText();
      //console.log("Text:", text);

      const parothers = await parseParothers(text);
      if (Object.keys(parothers).length > 0) {
        parsedSections.push(await parseParothers(text));
      } else {
        console.warn("Did not parse because missing Parother key:", text);
      }
    }

    const mergedParsedSections = parsedSections.reduce((acc, section) => {
      return { ...acc, ...section };
    }, {});

    console.log("Merged parsed sections:", mergedParsedSections);

    let word = {
      ...wordEl,
      wordType,
      wordDefinitionFeatures,
      etyhist: mergedParsedSections["etyhist"] || "",
      derives:
        mergedParsedSections["derives"]
          ?.split(",")
          .map((d: string) => d.trim().replace(".", "")) || null,
    } as WordEl;

    if (!word.completeWordForm) {
      word.completeWordForm = completeWordForm;
    }

    console.log(
      `Adding word ${word.wordForm} and its properties to database...`
    );
    await db.insertWord(word);
  }

  return words;
}

async function parseParothers(text: string): Promise<Record<string, string>> {
  const result: Record<string, string> = {};
  const sections: Sections = {
    prononciation: ["Prononc."],
    etyhist: ["Étymol. et Hist.", "Étymol. ET HIST.", "Étymol."],
    derives: ["Der. et composés"],
  };

  const sectionPattern = new RegExp(
    `(${Object.values(sections)
      .flatMap((arr) => arr)
      .join("|")})\\s*[.:−]?\\s*`,
    "g"
  );

  //console.log("Section pattern:", sectionPattern);

  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = sectionPattern.exec(text)) !== null) {
    const sectionName = Object.keys(sections).find((key: string) =>
      sections[key as SectionKey].includes(match![1])
    );

    if (!sectionName) {
      console.error("Could not find section name for match:", match);
      continue;
    }

    console.log("Section name:", sectionName);
    const startIndex = match.index + match[0].length;

    const nextMatch = sectionPattern.exec(text);
    sectionPattern.lastIndex = startIndex; // reset lastIndex for the next iteration

    const endIndex = nextMatch ? nextMatch.index : text.length;

    let sectionContent = text.substring(startIndex, endIndex).trim();

    result[sectionName] = sectionContent;

    lastIndex = endIndex;
  }

  return result;
}

async function getWordsOnPage(page: Page): Promise<WordEl[]> {
  function removeGenderSuffix(wordForm: string) {
    const suffixPattern = Object.entries(genderSuffixes)
      .map(([base, gendered]) => `(${base})-${gendered}$`)
      .join("|");

    const regex = new RegExp(suffixPattern);

    return wordForm.replace(regex, "$1");
  }

  const wordLinks: WordEl[] = [];
  let currentPage = 1;

  while (true) {
    console.log(`Scraping page ${currentPage}...`);

    let frameLocator = page.frameLocator("frame[name='fen1']");

    // wait a bit
    await page.waitForTimeout(3000);

    const isNextResultsButton = await frameLocator
      .locator("img[src='/dendien/ima/tlfiv4/suivants.gif']")
      .isVisible();

    console.log("Next results button visible:", isNextResultsButton);

    frameLocator = page.frameLocator("frame[name='fen2']");

    const tableEl = frameLocator.locator("table[cellspacing='3']");

    const rowEls = await tableEl.locator("tr").all();

    console.log(`Found ${rowEls.length} rows)`);

    // for each td in tr, find a and add to wordLinks
    for (const rowEl of rowEls) {
      console.log(`Treating row: ${rowEl} out of ${rowEls.length - 1}`);

      const completeWordForm = await rowEl
        .locator("td[bgcolor='#EOEOD0'] b")
        .innerText();
      console.log("completeWordForm:", completeWordForm);

      if (excludedWords.includes(completeWordForm)) {
        console.log("Excluded word:", completeWordForm);
        continue;
      }

      let wordForm = "";
      if (
        specialSplitWords.some((s) =>
          completeWordForm.split(",")[0].toLowerCase().includes(s)
        )
      ) {
        wordForm = completeWordForm.split(",")[0].toLowerCase();
      } else if (
        badlyFormattedWords.some((s) =>
          completeWordForm.split(",")[0].toLowerCase().includes(s)
        )
      ) {
        wordForm = completeWordForm
          .split(",")[0]
          .toLowerCase()
          .replace("subst.", "");
      } else {
        wordForm =
          // fixes issue with "un" and "une" at the beginning of the word
          // e.g. "un tribordais, subst. masc. (dans l'article -AIS, -AISE, -OIS, -OISE, suff.)"
          (
            ["un", "une"].includes(completeWordForm.split(" ")[0])
              ? completeWordForm.split(" ")[1]
              : completeWordForm.split(" ")[0]
          )
            .replace(/[.,\d\s]|\(\w+\)/g, "")
            // fixes words with hyphen in parentheses
            // e.g. "anti(-)impérialiste"
            .replace(/\((-)\)/g, "-")
            .toLowerCase();

        wordForm = removeGenderSuffix(wordForm);
      }

      console.log("wordForm:", wordForm);

      const url = "https://www.cnrtl.fr/definition/" + wordForm;
      console.log("url:", url);

      if (completeWordForm && wordForm && url) {
        wordLinks.push({
          completeWordForm,
          wordForm,
          url,
        });
      } else {
        console.error(
          `Could not find wordForm ${wordForm} or url ${url} for row: ${rowEl}`
        );
      }
    }

    if (!isNextResultsButton) {
      console.log("No 'Next' button found, stopping.");
      break;
    }

    console.log("Clicking 'Next' button...");

    frameLocator = page.frameLocator("frame[name='fen1']");

    // Click the 'Next' button to go to the next page
    await frameLocator
      .locator("img[src='/dendien/ima/tlfiv4/suivants.gif']")
      .click();

    // Wait for the next page to load
    await page.waitForLoadState("networkidle");

    currentPage++;
  }

  console.log("Finished scraping all pages.");
  console.log("wordLinks:", wordLinks);

  return wordLinks;
}

async function scrapeTLF(
  page: Page,
  wordsFile: string | null,
  recursiveSearch: boolean = false
): Promise<void> {
  const startTime = new Date().getTime();

  const selectOptions = {
    X1: "def",
    X2: "textdef",
    X3: "code",
    ZZZZ1: "p 2",
    ZZZZ2: "p 3",
  };

  const inputs = {
    YY2: ["&d1 &lnhumains", "&d1 celui", "&d1 celle", "&d1 quelqu'un"],
    YY3: "subst.",
  };

  console.log("Scraping TLF website...");

  let wordsOnPage = null;

  if (!wordsFile) {
    const wordEls: WordEl[][] = [];

    const answer = await askCreateWordFile();
    if (!["yes", "y"].includes(answer)) {
      console.log("Exiting...");
      process.exit(0);
    }

    // ask where to save the new words file
    const wordsFilePathRel = await new Promise<string>((resolve) => {
      rlInterface.question(
        "Enter the relative path to save the new words file: ",
        (input) => {
          resolve(input);
        }
      );
    });

    for (let i = 0; i < inputs.YY2.length; i++) {
      wordEls.push(
        await createWordFile(page, true, selectOptions, inputs, { index: i })
      );
    }

    // flatten and save to file
    wordsOnPage = wordEls.flat();
    fs.writeFileSync(wordsFilePathRel, JSON.stringify(wordsOnPage));

    process.exit(0);
  }

  if (recursiveSearch) {
    console.log("Launching recursive mode...");
    wordsOnPage = JSON.parse(fs.readFileSync(wordsFile, "utf-8")) as WordEl[];
    const db = new Database("dbs/recursive.db");

    for (const word of wordsOnPage) {
      const wordForm = word.wordForm;
      const wordEls = await createWordFile(page, false, selectOptions, inputs, {
        wordForm,
      });
      // save to new db
      for (let wordEl of wordEls) {
        wordEl = {
          ...wordEl,
          parentCompleteWord: word.completeWordForm,
          parentWord: word.wordForm,
        };
        db.insertWord(wordEl, true);
      }
    }

    process.exit(0);
  }

  //if wordsFile is provided, read the file and scrape the words
  if (wordsFile && wordsFile.length > 0) {
    console.log(`Reading words from file: ${wordsFile}`);
    if (wordsFile.split(".").pop() == "json") {
      wordsOnPage = JSON.parse(fs.readFileSync(wordsFile, "utf-8")) as WordEl[];
    } else if (wordsFile.split(".").pop() == "csv") {
      const json = csv.fieldDelimiter(",").getJsonFromCsv(wordsFile);
      wordsOnPage = json.map(
        (item) =>
          ({
            completeWordForm: "",
            wordForm: item.word,
            url: "",
          } as WordEl)
      );
    } else {
      console.error("Invalid file format. Please provide a JSON or CSV file.");
      process.exit(1);
    }
  }

  console.log(`wordsOnPage: ${wordsOnPage}`);

  if (wordsOnPage) {
    console.log("Scraping words on page...");
    const results = await scrapeWordPage(page, wordsOnPage);

    if (!results) {
      console.error("No results found.");
      process.exit(1);
    }

    const endTime = new Date().getTime();
    const elapsedTimeMinutes = (endTime - startTime) / 60000;

    console.log(`Scraping completed in ${elapsedTimeMinutes} minutes.`);
  } else {
    console.error("No words to scrape.");
  }
}

async function createWordFile(
  page: Page,
  createWordList: boolean = false,
  selectOptions: any,
  inputs: any,
  options?: { index?: number; wordForm?: string }
): Promise<WordEl[]> {
  await page.goto("http://stella.atilf.fr/", { waitUntil: "networkidle" });

  let frameLocator = page.frameLocator(
    "frame[src='http://atilf.atilf.fr/scripts/mep.exe?CRITERE=ACCUEIL_TLFI;MENU=menu_tlfi;ONGLET=tlfi;OO1=1;OO2=1;ISIS=isis_tlfi.txt;OUVRIR_MENU=0;']"
  );

  // click input class BoutonRessource
  await frameLocator.locator("input[class='BoutonRessource']").click();

  if (createWordList) {
    await frameLocator
      .locator("img[src='/dendien/ima/tlfiv4/listes.gif']")
      .click();

    await submitWordList(page);
  }

  console.log("Accessing search page...");

  // !watch out: this may not work with createWordList
  // ! we may need to change to page.locator?
  // ! to test
  await frameLocator
    .locator("img[src='/dendien/ima/tlfiv4/rechcomp.gif']")
    .click();

  frameLocator = page.frameLocator("frame[name='fen2']");

  console.log("Selecting selects...");
  for (const [selectName, optionValue] of Object.entries(selectOptions)) {
    console.log(`Selecting ${selectName} with option ${optionValue}`);
    const selectSelector = `tbody tr td select[name="${selectName}"]`;

    await frameLocator.locator(selectSelector).selectOption({
      value: optionValue,
    });

    console.log(`Selected ${selectName} with option ${optionValue}`);
  }

  console.log("Filling inputs...");
  for (const [inputName, inputValue] of Object.entries(inputs)) {
    let value = inputValue;

    if (inputName === "YY2") {
      if (Array.isArray(inputValue) && options && options.index !== undefined) {
        value = inputValue[options.index];
      } else if (options && options.wordForm) {
        // only search first word, too many false positives otherwise
        value = "&d0 " + options.wordForm;
      }
    }

    console.log(`Filling input ${inputName} with value ${value}`);
    const inputSelector = `tbody tr td input[name="${inputName}"]`;

    await frameLocator.locator(inputSelector).fill(value);
  }

  // click on input type submit which text contains "Valider"
  console.log("Clicking submit button...");
  await frameLocator.locator('input[type="submit"][value*="Valider"]').click();

  // wait for the page to load
  await page.waitForLoadState("networkidle");

  console.log(`Accessed page: ${page.url()}`);

  //code for result page
  frameLocator = page.frameLocator("frame[name='fen1']");

  const isResultsVisible = await frameLocator
    .getByText("Résultats")
    .isVisible();

  if (!isResultsVisible) {
    console.warn("Only found one result, skipping...");
    return [];
  }

  const resultsText = await frameLocator.getByText("Résultats").innerText();

  console.log("Results text:", resultsText);

  let resultsTotal = 0;
  let resultsFrom = 0;
  let resultsTo = 0;

  const regex = /Résultats (\d+) à (\d+)\/(\d+)/;
  const match = resultsText.match(regex);

  if (match) {
    resultsFrom = parseInt(match[1]);
    resultsTo = parseInt(match[2]);
    resultsTotal = parseInt(match[3]);
  } else {
    console.error("Could not parse results text:", resultsText);
  }

  console.log("Results:", resultsFrom, resultsTo, resultsTotal);

  return getWordsOnPage(page);
}

async function askCreateWordFile(): Promise<string> {
  return new Promise<string>((resolve) => {
    rlInterface.question(
      "No words file provided. Do you want to scrape words from page? (Y/N): ",
      (input) => {
        resolve(input.toLowerCase());
      }
    );
  });
}

async function main(): Promise<void> {
  const program = new Command();

  program
    .name("tlfi_scraper")
    .version("0.1.0")
    .description(
      "Scrape the TLF website for word definitions and other linguistic features."
    )
    .option(
      "-w, --words <file>",
      "File containing the words to scrape. Can be JSON or CSV (see examples in /words folder). If no word file is provided, a predefined 'human noun search' will first be made in order to retrieve relevant words to scrape and add them to a JSON file."
    )
    .option("-r, --recursive", "Recursive search")
    .option("-h, --help", "Show help")
    .parse();

  const options = program.opts();

  const wordsFile = options.words;
  const recursive = options.recursive ? true : false;

  const browser = await chromium.launch();
  const page = await browser.newPage();

  if (options.help) {
    program.help();
    process.exit(0);
  }

  try {
    await scrapeTLF(page, wordsFile, recursive);
    process.exit(0);
  } catch (error) {
    console.error("An error occurred:", error);
  } finally {
    await browser.close();
  }
}

main();
