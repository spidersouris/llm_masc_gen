type SectionKey = "prononciation" | "etyhist" | "derives";
type Sections = Record<SectionKey, string[]>;

interface WordEl {
  completeWordForm: string;
  wordForm: string;
  wordType?: string;
  wordDefinitionFeatures?: WordDefinitionFeatures[];
  etyhist?: string;
  derives?: string[];
  url: string;
  parentCompleteWord?: string;
  parentWord?: string;
}

interface WordDefinitionFeatures {
  definition: string;
  bracketText: string;
  indicator: string;
  domain: string;
  phrase: string;
  examples: DefinitionExample[];
}

interface DefinitionExample {
  example: string;
  metadata: DefinitionExampleMetadata;
}

interface DefinitionExampleMetadata {
  author: string;
  title: string;
  date: string;
}

export {
  SectionKey,
  Sections,
  WordEl,
  WordDefinitionFeatures,
  DefinitionExample,
};
