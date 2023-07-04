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
	model: 'gpt-4',
	openAIApiKey: process.env.OPENAI_API_KEY
};

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const directoryPath = (directoryName) => path.resolve(currentDir, `../data/docs/${directoryName}`);
const outputPath = (fileName) => path.resolve(currentDir, `${fileName}`);
const keyPath = (directoryName) => path.resolve(currentDir, `../data/text/${directoryName}/data.key.json`);

const buildModel = () => {
	return new ChatOpenAI({
		temperature: 0.2,
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
	const query = "What is a Slug in SvelteKit?";
	const directoryName = 'SvelteKit';
	const result = await callCodeChain(query, directoryName);
	console.log(result.text);
	console.log(result.urls);
};

async function callCodeChain(query, directoryName) {
	const retriever = await retrieverFromDirectory(directoryName, 5);
	const results = await retriever.getRelevantDocuments(query);
	const metadata = results.map(result => ({'key': result.metadata.key, 'url': result.metadata.url}));
	let uniqueURLs = [...new Set(metadata.map(item => item.url))];
	uniqueURLs = `Sources: ${uniqueURLs.join(', ')};`
	let relevantDocuments = await getContentFromKeys(metadata, directoryName);
	relevantDocuments = relevantDocuments.join('\n');

	const SystemTemplate = 'You are a Chatbot who answers user questions about documentation for a given framework or programming language. Please use only the users provided documentation snippets to answer their query.';
	const HumanTemplate = 'Please answer the following user question: "{query}", using the documentation snippets provided: {relevantDocuments}. Please also include examples in the response if relevant';
	const codeChainPrompt = ChatPromptTemplate.fromPromptMessages([
		SystemMessagePromptTemplate.fromTemplate(SystemTemplate),
		HumanMessagePromptTemplate.fromTemplate(HumanTemplate)
	]);
	const codeChain = new LLMChain({llm: buildModel(), prompt: codeChainPrompt});
	const result = await codeChain.call({query: query, relevantDocuments: relevantDocuments});
	result.urls = uniqueURLs;
	return result;
}


const getContentFromKeys = (keys, directory) => {
	const keyFile = fs.readFileSync(keyPath(directory), 'utf8');
	const keyData = JSON.parse(keyFile);

	return keys.map(key => keyData[key.key]);
}


run();
