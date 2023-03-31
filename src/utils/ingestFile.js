import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import * as path from 'path';
import { fileURLToPath } from 'url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));

const fileName = "GameDesignPatterns.txt";
const outputDir = 'dartDocs'

const ingestedPath = path.resolve(currentDir, `../data/text/${fileName}`);
const docsPath = path.resolve(currentDir, `../data/docs/${outputDir}`);


export const run = async () => {
    const text = fs.readFileSync(ingestedPath, "utf8");
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 100 });
    const docs = await splitter.createDocuments([text]);
    const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
    await vectorStore.save(docsPath);

};

run();
