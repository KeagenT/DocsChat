import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import * as path from 'path';
import { fileURLToPath } from 'url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const docsPath = path.resolve(currentDir, '../data/docs');


export const run = async () => {
    const vectorStore = await HNSWLib.load(docsPath, new OpenAIEmbeddings());

};

run();
