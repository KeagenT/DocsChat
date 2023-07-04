import { HNSWLib } from 'langchain/vectorstores/hnswlib';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { LLMChain, RetrievalQAChain } from 'langchain/chains';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import {
	ChatPromptTemplate,
	HumanMessagePromptTemplate,
	SystemMessagePromptTemplate
} from 'langchain/prompts';

// Default GPT model to use and OpenAI API key
// This can be more configurable by loading from a .env file instead of your .bashrc etc file
const DEFAULT_MODEL = {
	model: 'gpt-3.5-turbo',
	openAIApiKey: process.env.OPENAI_API_KEY
};

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const directoryPath = (directoryName) => path.resolve(currentDir, `../data/docs/${directoryName}`);
const outputPath = (fileName) => path.resolve(currentDir, `${fileName}`);
const keyPath = (directoryName) => path.resolve(currentDir, `../data/text/${directoryName}/data.key.json`);

const buildModel = () => {
	return new ChatOpenAI({
		temperature: 0,
		...DEFAULT_MODEL
	});
};

const buildRetrievalChain = async (directoryName) => {
	const vectorStore = await HNSWLib.load(directoryPath(directoryName), new OpenAIEmbeddings());
	return RetrievalQAChain.fromLLM(buildModel(), vectorStore.asRetriever(), {returnSourceDocuments: true});
};

const retrieverFromDirectory = async (directoryName, results) => {
	const vectorStore = await HNSWLib.load(directoryPath(directoryName), new OpenAIEmbeddings());
	return vectorStore.asRetriever(results);
}

export const run = async () => {
	const query = "What's is the difference between a page.server.js and page.js file in SvelteKit?";
	const retriever = await retrieverFromDirectory('SvelteKit', 5);
	const results = await retriever.getRelevantDocuments(query);
	const metadata = results.map(result => ({'key': result.metadata.key, 'url': result.metadata.url}));
	const uniqueURLs = [...new Set(metadata.map(item => item.url))];
	console.log(metadata);
	console.log(...getContentFromKeys(metadata, 'SvelteKit'))
};

const getContentFromKeys = (keys, directory) => {
	const keyFile = fs.readFileSync(keyPath(directory), 'utf8');
	const keyData = JSON.parse(keyFile);

	return keys.map(key => keyData[key.key]);
}


run();
