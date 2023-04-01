import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import * as fs from "fs";
import * as path from 'path';
import { fileURLToPath } from 'url';
import { LLMChain, RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models";
import { ChainTool } from "langchain/tools";
import { initializeAgentExecutor } from "langchain/agents";
import { ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate } from "langchain/prompts";

// Default GPT model to use and OpenAI API key
// This can be more configurable by loading from a .env file instead of your .bashrc etc file
const DEFAULT_MODEL = {
    model: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
};

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const directoryPath = (directoryName) => path.resolve(currentDir, `../data/docs/${directoryName}`);
directoryPath("gameDesignBook");

const buildModel = () => {
    return new ChatOpenAI({
        ...DEFAULT_MODEL
    });
}

const buildRetrievalChain = async (directoryName) => { 
    const vectorStore = await HNSWLib.load(directoryPath(directoryName), new OpenAIEmbeddings());
    return RetrievalQAChain.fromLLM(buildModel(), vectorStore.asRetriever());
}

export const run = async () => {
    const bookChain = await buildRetrievalChain("gameDesignBook");
    const bookTool = new ChainTool({
        name: 'game-design-patterns-tool',
        description: 'Game Design Patterns Book QA - used to ask questions about the book Game Design Patterns by Robert Nystrom',
        chain: bookChain,
    });
    const dartDocsChain = await buildRetrievalChain("dartDocs");

    const dartDocsTool = new ChainTool({
        name: 'dart-docs-tool',
        description: 'Dart Docs QA - used to ask questions about the Dart programming language by directly searching the Dart documentation',
        chain: dartDocsChain,
    });

    const humanTemplate = "The explanation to convert is as follows: {explanation}";
    const systemCodingTemplate = "You are a Senior Software Engineer who specializes in Dart. You are able to convert explanations of algorithms, design patterns, or snippets from another language, into Dart code examples.";
    const codingPrompt = ChatPromptTemplate.fromPromptMessages([
        SystemMessagePromptTemplate.fromTemplate(systemCodingTemplate),
        HumanMessagePromptTemplate.fromTemplate(humanTemplate),
      ]);
    const codingChain = new LLMChain({ llm: buildModel(), prompt: codingPrompt });
    const codingTool = new ChainTool({
        name: 'coding-tool',
        description: 'Coding Tool - used to generate Dart code examples from input conceptual explanations',
        chain: codingChain,
    });

    
    const bookExecutor = await initializeAgentExecutor([bookTool, codingTool, dartDocsTool], buildModel(), "chat-zero-shot-react-description");
    const input = "Please show me how to implement the state pattern with code examples and explain the state pattern in the context of video games conceptually.";
    const result = await bookExecutor.call({input}); 
    const JSONString = JSON.stringify(result, null, 2);
    fs.writeFileSync('output.json', JSONString, 'utf8');
    console.log(JSONString);

};

run();
