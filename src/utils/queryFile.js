import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import * as path from 'path';
import { fileURLToPath } from 'url';
import { OpenAI } from "langchain";
import { ConversationalRetrievalQAChain } from "langchain/chains";

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const docsPath = path.resolve(currentDir, '../data/docs');


export const run = async () => {
    const vectorStore = await HNSWLib.load(docsPath, new OpenAIEmbeddings());
    const model = new OpenAI({
        model: "gpt-3.5-turbo",
        openAIApiKey: process.env.OPENAI_API_KEY,
    });
    const chain = ConversationalRetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
    const question = "What are some of the problems the command pattern is most apt to solve?";
    const response = await chain.call({question, chat_history: []});
    console.log(response);
    const chatHistory = question + response.text;
    const followUpResponse = await chain.call({
        question: "What benefits does the Command pattern provide?",
        chat_history: chatHistory,
    });
    console.log(followUpResponse);
};

run();
