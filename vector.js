import { config } from "dotenv";
config();

import { ChatOpenAI } from "@langchain/openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";


const apiKey = process.env.OPENAI_API_KEY;

const chatModel = new ChatOpenAI({apiKey});

const loader = new CheerioWebBaseLoader(
  "https://blogs.webmd.com/diet-nutrition/20200305/what-to-eat-to-boost-your-immune-system"
);

const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter();

const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings();

const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt,
});

const retriever = vectorstore.asRetriever();

const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});

const result = await retrievalChain.invoke({
    input: "What to eat to boost your immune system?",
  });
  
  console.log(result.answer);
