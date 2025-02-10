import { WordEl } from "./types";
import sqlite3 from "sqlite3";
import { Database as SQLiteDatabase } from "sqlite3";

export class Database {
  private db: SQLiteDatabase;

  constructor(dbPath: string = "dbs/db_tlfi.db") {
    this.db = new sqlite3.Database(dbPath, (err) => {
      if (err) {
        console.error("Error opening database", err);
      } else {
        console.log("Database connected");
        this.initializeTables(dbPath);
      }
    });
  }

  private initializeTables(dbPath: string) {
    let createWordTable = `
    CREATE TABLE IF NOT EXISTS words (
      id INTEGER PRIMARY KEY,
      completeWordForm TEXT,
      wordForm TEXT,
      wordType TEXT,
      wordDefinitionFeatures TEXT, -- Save as JSON
      etyhist TEXT,
      derives TEXT, -- Save as JSON
      url TEXT
    )`;

    if (dbPath.includes("recursive")) {
      createWordTable = `
      CREATE TABLE IF NOT EXISTS words (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        completeWordForm TEXT,
        wordForm TEXT,
        wordType TEXT,
        wordDefinitionFeatures TEXT, -- Save as JSON
        etyhist TEXT,
        derives TEXT, -- Save as JSON
        url TEXT,
        parentCompleteWord TEXT,
        parentWord TEXT
      )`;
    }

    this.db.run(createWordTable, (err) => {
      if (err) {
        console.error("Error creating table", err);
      }
    });
  }

  insertWord(word: WordEl, parentWord: boolean = false): Promise<void> {
    return new Promise((resolve, reject) => {
      let insertQuery = `
      INSERT INTO words (
        completeWordForm,
        wordForm,
        wordType,
        wordDefinitionFeatures,
        etyhist,
        derives,
        url
      ) VALUES (?, ?, ?, ?, ?, ?, ?)`;

      if (parentWord) {
        insertQuery = `
        INSERT INTO words (
          completeWordForm,
          wordForm,
          wordType,
          wordDefinitionFeatures,
          etyhist,
          derives,
          url,
          parentCompleteWord,
          parentWord
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`;
      }

      const wordDefinitionFeatures = word.wordDefinitionFeatures
        ? JSON.stringify(word.wordDefinitionFeatures) // Convert array to JSON
        : null;

      const derives = word.derives
        ? JSON.stringify(word.derives) // Convert array to JSON
        : null;

      console.log(word);

      this.db.run(
        insertQuery,
        [
          word.completeWordForm,
          word.wordForm,
          word.wordType || null,
          wordDefinitionFeatures,
          word.etyhist || null,
          derives,
          word.url,
          //word.parentCompleteWord || null,
          //word.parentWord || null,
        ],
        (err) => {
          if (err) {
            console.error(`Error inserting word ${word.wordForm}:`, err);
            reject(err);
          } else {
            console.log(`Word ${word.wordForm} inserted successfully`);
            resolve();
          }
        }
      );
    });
  }
}
