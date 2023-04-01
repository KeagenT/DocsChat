# Docs Chat
Docs Chat ~~is a UI interface~~ will be a UI interface for [langchain's](https://hwchase17.github.io/langchainjs/docs/overview) document loader
embedding wrapper, used to fit Documentation, Books, or any other kind of "Document" into retrieval context for an LLM.
I'm using it here with [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5) to "talk" with documentation I don't really
understand. 
Usage currently requires a developer's understanding of Node projects and environment variables.

## Usage
### With A Non-existent incomplete UI
This SvelteKit project uses `node version 18` as a dependent version for langchainJS.
Usage follows standard convention for other SvelteKit projects.

Install dependencies needed to run this project with

```bash
 npm install
```
and run this project with

```bash
 npm run dev
```
### In its current state

This project *requires* an OpenAI API key to run. 
[You can find API Key information here](https://platform.openai.com/docs/api-reference).
In this particular hardcoded environment,
I'm referencing the API key from my own .bashrc file inside queryFile.js,
and as such this assumes your terminal has an exported OPENAI_API_KEY variable.

Currently, this project's whole functionality is limited to the src/utils/ingestFile.js file,
and src/utils/queryFile.js files. the ingestFile.js file is used to convert a text file into an
embedding that can be used by langchainJS.

You can load a file to ask a model questions about with
```bash
npm run ingest <file-name> <output-directory-name>
```
assuming you've placed the file in the `src/data/text` directory, and `<file-name>` is the name of the file you want to embed,
and `<output-directory-name>` is the name of the directory you want to save the embedding within the data/docs directory.
This only needs to be done once per file, and the embedding will be saved in the data/docs directory for future use.

#### **Important Disclaimer**
*Ingesting a file uses OpenAI's embedding API, and as such, will cost you literal actual money, likely more than you'd expect from regular conversational usage of a model. e.g. embedding an entire book to test this on cost me roughly $0.40*

The queryFile.js file is a hard coded example of how to use langchainJS 
to ask questions about your embedded text file. Modify the prompts as you see fit,
change the relevant file names/directory names, and see the output with node src/utils/queryFile.js in your terminal.



