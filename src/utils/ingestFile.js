import { HNSWLib } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));

export const run = async (fileName, outputDir) => {
	const ingestedPath = path.resolve(currentDir, `../data/text/${fileName}`);
	const docsPath = path.resolve(currentDir, `../data/docs/${outputDir}`);

	const text = fs.readFileSync(ingestedPath, 'utf8');
	// Splits the input text file into smaller chunks that fit in your LLM's context
	const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 250, chunkOverlap: 50 });
	const docs = await splitter.createDocuments([text]);
	// Creates embeddings from the split text chunks
	const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
	// saves the embeddings to the output directory
	await vectorStore.save(docsPath);
};

const filename_input = process.argv[2];
const directoryOutput = process.argv[3];
if (filename_input && directoryOutput) {
	run(filename_input, directoryOutput);
} else {
	console.log(
		'Please provide a filename and output directory e.g. \n npm run ingest <filename> <output directory>'
	);
}
