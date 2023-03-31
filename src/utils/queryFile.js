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

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const docsPath = path.resolve(currentDir, '../data/docs/gameDesignBook');
const DEFAULT_MODEL = {
    model: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
};

const buildModel = () => {
    return new ChatOpenAI({
        ...DEFAULT_MODEL
    });
}

export const run = async () => {
    const vectorStore = await HNSWLib.load(docsPath, new OpenAIEmbeddings());
    const retrievalChain = RetrievalQAChain.fromLLM(buildModel(), vectorStore.asRetriever());
    const bookTool = new ChainTool({
        name: 'game-design-patterns-tool',
        description: 'Game Design Patterns Book QA - used to ask questions about the book Game Design Patterns by Robert Nystrom',
        chain: retrievalChain,
    });
    const humanTemplate = "The explanation to convert is as follows: {explanation}";
    const systemCodingTemplate = "You are a Senior Software Engineer who specializes in Dart. You are able to convert explanations of algorithms or design patterns into Dart code examples.";
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

    
    const bookExecutor = await initializeAgentExecutor([bookTool, codingTool], buildModel(), "chat-zero-shot-react-description");
    const input = "How do I conceptually implement the observer pattern based on the game design patterns book? After explaining the concept, please generate a Dart Code example of the pattern in addition to the conceptual explanation.";
    const result = await bookExecutor.call({input}); 
    console.log(`${JSON.stringify(result, null, 2)}`);

};

run();
