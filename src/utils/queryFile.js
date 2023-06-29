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
	// const svelteKitDocsChain = await buildRetrievalChain("svelteKitDocs");
	const svelteKitRetriever = await retrieverFromDirectory("svelteKitDocs", 10);

	// const targetLanguage = 'dart';
	// const targetContext = 'Video Game';
	// const bookQuestion = 'What is the State Design Pattern?';
	// const summaryResult = await getExplanationWithCodeSnippet(
	// 	bookQuestion,
	// 	targetLanguage,
	// 	targetContext
	// );
	const query = 'How do I submit data with a form';
	const question = `${query} in SveteKit?`;
	const relevantDocs = await svelteKitRetriever.getRelevantDocuments(query);
	const usefulResults = relevantDocs.filter((result) => helpsAnswerQuestionChain(result.pageContent, query));
	const explanationResults = usefulResults.filter((result) => !isCodeSnippetChain(result.pageContent));
	const codeResults = usefulResults.filter((result) => isCodeSnippetChain(result.pageContent));
	console.log("Useful docs", usefulResults);
	console.log("Explanation docs", explanationResults);
	console.log("Code docs", codeResults);
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

async function isCodeSnippetChain(input) {
	const criteria = 'Is the input a code snippet? e.g. `Print("Hello World")` and NOT "hello `world`\n"';
	const result = await evaluateCriteraChain(input, criteria);
	return result;
}

async function helpsAnswerQuestionChain(input, query) {
	const criteria = `Does this input help answer the question "${query}"?`;
	const result = await evaluateCriteraChain(input, criteria);
	return result;
}

async function evaluateCriteraChain(input, criteria) {
	const systemEvaluateTemplate = 'You are an NLP assistant who evaluates user input against a criteria question: {criteria} You are to return a boolean value of true or false. Respond ONLY with this value and no other explanantion you must be absolutely certain it is true, or absolutely certain it is false. e.g. user: "print(`hello world`)" assistant: "true", user: "hello world" assistant: "false"';
	const humanEvaluateTemplate = 'The input is as follows: {input} {criteria} Make sure the response is only a boolean value of true or false.';
	const evaluateCriteriaPrompt = ChatPromptTemplate.fromPromptMessages([ SystemMessagePromptTemplate.fromTemplate(systemEvaluateTemplate), HumanMessagePromptTemplate.fromTemplate(humanEvaluateTemplate)]);
	const evaluateCriteriaChain = new LLMChain({ llm: buildModel(), prompt: evaluateCriteriaPrompt });
	const result = await evaluateCriteriaChain.call({ input: input, criteria: criteria });
	const resultBool = result.text.toLowerCase() === 'true' ? true : false;
	return resultBool;
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
