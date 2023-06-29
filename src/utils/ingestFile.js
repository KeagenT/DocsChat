import { HNSWLib } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));

export const run = async (fileName, outputDir, inputType) => {
	const ingestedPath = path.resolve(currentDir, `../data/text/${fileName}`);
	const docsPath = path.resolve(currentDir, `../data/docs/${outputDir}`);

	const text = fs.readFileSync(ingestedPath, 'utf8');
	if (inputType === 'text') {
		parseText(text, docsPath);
	}
	if (inputType === 'json') {
		parseJSON(text, docsPath);
	}
};

export const parseText = async (text, docsPath) => {
	const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 250, chunkOverlap: 50 });
	const docs = await splitter.createDocuments([text]);
	// Creates embeddings from the split text chunks
	const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
	// saves the embeddings to the output directory
	await vectorStore.save(docsPath);
}

// arbitrary JSON format for web pages I've scraped
// e.g. {pages: [{url: "https://www.google.com", content: "This is the content of the page"}, ...]]}
// Want to see if this can be used to add sources to retrieval chains with the sources being the MetaData URL
const parseJSON = async (text, docsPath) => {
	const input = await JSON.parse(text);
	const pages = input.pages;
	const content = [...pages.map(page => page.content)];
	const metadata =  [...pages.map(page => ({url: page.url}))]
	const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 250, chunkOverlap: 50 });
	const docs = await splitter.createDocuments(content, metadata);
	const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
	await vectorStore.save(docsPath);
}

const args = process.argv.slice(2);
const options = {};
const INPUT_FLAG = '--input-type';

while (args.length) {
  const arg = args.shift();
  if (arg === INPUT_FLAG) {
    options.inputType = args.shift();
  } else if (!options.filename) {
    options.filename = arg;
  } else if (!options.directoryOutput) {
    options.directoryOutput = arg;
  }
}

const { filename, directoryOutput, inputType } = options;

if (filename && directoryOutput && inputType) {
  run(filename, directoryOutput, inputType);
} else {
  console.log(
    'Please provide a filename, output directory, and input type e.g. \n npm run ingest <filename> <output directory> -- --input-type <input type>'
  );
  console.log(options);
}

