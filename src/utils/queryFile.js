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
	

	
};



run();
