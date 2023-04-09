import { HNSWLib } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { LLMChain, RetrievalQAChain } from 'langchain/chains';
import { ChatOpenAI } from 'langchain/chat_models';
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
		...DEFAULT_MODEL
	});
};

const buildRetrievalChain = async (directoryName) => {
	const vectorStore = await HNSWLib.load(directoryPath(directoryName), new OpenAIEmbeddings());
	return RetrievalQAChain.fromLLM(buildModel(), vectorStore.asRetriever());
};

export const run = async () => {
	// const bookChain = await buildRetrievalChain("gameDesignBook");
	const dartDocsChain = await buildRetrievalChain("dartDocs");

	const targetLanguage = 'dart';
	const targetContext = 'Video Game';
	const bookQuestion = 'What is the State Design Pattern?';
	// const summaryResult = await getExplanationWithCodeSnippet(
	// 	bookQuestion,
	// 	targetLanguage,
	// 	targetContext
	// );
    const summaryResult = await dartDocsChain.call({ query: "How do I make a variable in Dart?" });
	console.log(summaryResult.text);
	await fs.promises.writeFile(outputPath('output.txt'), summaryResult.text);
};

async function callBookChain(question) {
	const bookChain = await buildRetrievalChain('gameDesignBook');
	const result = await bookChain.call({ query: question });
	return result;
}

async function callCodeChain(userExplanation, inputLanguage, inputContext) {
	const humanTemplate =
		'The explanation or snippet to convert is as follows: {explanation}.  You output ONLY markdown blocks of commented {language} code e.g. ```{language}\nExample Code Goes here```. Remember to contextualize the example in the style of {context} development.';
	const systemCodingTemplate =
		'You are a Senior Software Engineer who specializes in {language}. You are able to convert explanations of algorithms, design patterns, or snippets from another language, into {language} code examples. You always try to contextualize code examples with usage in the style of {context} development for maximum understanding.';
	const codingPrompt = ChatPromptTemplate.fromPromptMessages([
		SystemMessagePromptTemplate.fromTemplate(systemCodingTemplate),
		HumanMessagePromptTemplate.fromTemplate(humanTemplate)
	]);
	const codingChain = new LLMChain({ llm: buildModel(), prompt: codingPrompt });
	const result = await codingChain.call({
		explanation: userExplanation,
		language: inputLanguage,
		context: inputContext
	});
	return result;
}

async function codeSummaryChain(userInputCode, userInputExplanation) {
	const systemSummaryTemplate =
		'You are a helpful Assistant, who reformats input code snippets and explanations related to code snippets. You are to take both a markdown code snippet, and a user explanation, summarize the explanation and append it to the bottom of the codeblock in the following format: ```User Code Snippet Here```\n Summarized Explanation Here';
	const humanSummaryTemplate =
		'The code snippet is as follows: {code}. The explanation is as follows: {explanation}. Make sure the summarized explanation and original code snippet follow this formatting: ```User Code Snippet Here```\n Summarized Explanation Here';
	const summaryPrompt = ChatPromptTemplate.fromPromptMessages([
		SystemMessagePromptTemplate.fromTemplate(systemSummaryTemplate),
		HumanMessagePromptTemplate.fromTemplate(humanSummaryTemplate)
	]);
	const summaryChain = new LLMChain({ llm: buildModel(), prompt: summaryPrompt });
	const result = await summaryChain.call({
		code: userInputCode,
		explanation: userInputExplanation
	});
	return result;
}

async function getExplanationWithCodeSnippet(question, targetLanguage, targetCodeContext) {
	const bookResult = await callBookChain(question);
	const codeResult = await callCodeChain(bookResult.text, targetLanguage, targetCodeContext);
	const summaryResult = await codeSummaryChain(codeResult.text, bookResult.text);
	return summaryResult;
}

run();
