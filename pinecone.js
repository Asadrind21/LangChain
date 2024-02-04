import { config } from "dotenv";
config();

import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Pinecone } from "@pinecone-database/pinecone";
import { AttributeInfo } from "langchain/schema/query_constructor";
import { OpenAIEmbeddings, OpenAI } from "@langchain/openai";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";
import { PineconeTranslator } from "langchain/retrievers/self_query/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const apiKey = process.env.OPENAI_API_KEY;

const chatModel = new OpenAI({ apiKey });

if (
  !process.env.PINECONE_API_KEY ||
  !process.env.PINECONE_ENVIRONMENT ||
  !process.env.PINECONE_INDEX
) {
  throw new Error(
    "PINECONE_ENVIRONMENT and PINECONE_API_KEY and PINECONE_INDEX must be set"
  );
}

const pinecone = new Pinecone();

const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);
// Load PDF
const loader = new PDFLoader("book.pdf" , {
  parsedItemSeparator: "",
});

const book = await loader.load();

// console.log(book.length);
// console.log(book[0].pageContent.length);


const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const docOutput = await splitter.splitDocuments([
  new Document({ pageContent: book }),
]);

//console.log(docOutput);

const embeddings = new OpenAIEmbeddings();
const llm = new OpenAI();
const documentContents = "Book about disease diagnosis";
const vectorStore = await PineconeStore.fromDocuments(docOutput, embeddings, {
  pineconeIndex: pineconeIndex,
});

const selfQueryRetriever = await SelfQueryRetriever.fromLLM({
  llm,
  vectorStore,
  documentContents,
  /**
   * We need to create a basic translator that translates the queries into a
   * filter format that the vector store can understand. We provide a basic translator
   * translator here, but you can create your own translator by extending BaseTranslator
   * abstract class. Note that the vector store needs to support filtering on the metadata
   * attributes you want to query on.
   */
  structuredQueryTranslator: new PineconeTranslator(),
});

const query1 = await selfQueryRetriever.getRelevantDocuments(
  "I am sick?"
);

console.log(query1);