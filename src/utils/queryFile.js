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
import { CallbackManager } from "langchain/callbacks";

// Default GPT model to use and OpenAI API key
// This can be more configurable by loading from a .env file instead of your .bashrc etc file
const DEFAULT_MODEL = {
    model: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
};

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const directoryPath = (directoryName) => path.resolve(currentDir, `../data/docs/${directoryName}`);
const outputPath = (fileName) => path.resolve(currentDir, `${fileName}`);
directoryPath("gameDesignBook");

const replaceNewLines = (str) => str.replace(/\\n/g, '\n');

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
        description: 'Game Design Patterns Book QA - Book Used to ask questions about coding design patterns in the context of video games',
        chain: bookChain,
    });
    const dartDocsChain = await buildRetrievalChain("dartDocs");

    const dartDocsTool = new ChainTool({
        name: 'dart-docs-tool',
        description: 'Dart Docs QA - used to ask questions about the Dart programming language by directly searching the Dart documentation',
        chain: dartDocsChain,
    });

    const humanTemplate = "The explanation to convert is as follows: {explanation}";
    const systemCodingTemplate = "You are a Senior Software Engineer who specializes in Dart. You are able to convert explanations of algorithms, design patterns, or snippets from another language, into Dart code examples. You also receive JSON string signifying chat action history, this is largely irrelevant to fulfilling your purpose.";
    const codingPrompt = ChatPromptTemplate.fromPromptMessages([
        SystemMessagePromptTemplate.fromTemplate(systemCodingTemplate),
        HumanMessagePromptTemplate.fromTemplate(humanTemplate),
      ]);
    const codingChain = new LLMChain({ llm: buildModel(), prompt: codingPrompt });
    const codingTool = new ChainTool({
        name: 'coding-tool',
        description: 'Coding Tool - used to generate code examples from explanations obtained from books or other sources',
        chain: codingChain,
    });

    CallbackManager
    const callbackManager = CallbackManager.fromHandlers({
        async handleAgentAction(action) {
          console.log("handleAgentAction", action);
        },
      });

    const bookExecutor = await initializeAgentExecutor([bookTool, codingTool, dartDocsTool], buildModel(), "chat-zero-shot-react-description", callbackManager);
    const input = "Please show me how to implement the state pattern with code examples and explain the state pattern in the context of video games conceptually. The presence of a code sample in the final output summary from intermediate steps is extremely important.";
    const result = await bookExecutor.call({input}); 
    const JSONResult = JSON.stringify(result, null, 2);
    console.log(JSONResult);
    await fs.promises.writeFile(outputPath("output.txt"), JSONResult);

    

    

};

run();
