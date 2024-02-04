import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAI } from "langchain/llms/openai";
import { PromptTemplate } from "langchain/prompts";
import { loadSummarizationChain } from "langchain/chains";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { BufferMemory, ChatMessageHistory } from "langchain/memory";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { LocalStorage } from "node-localstorage";
import localforage from "localforage";
import fakeIndexedDB from "fake-indexeddb";
import FDBKeyRange from "fake-indexeddb/lib/FDBKeyRange";
import { promisify } from "util";
import fs from "fs";
import Datastore from "nedb";
import { HumanMessage, AIMessage } from "langchain/schema";

import { config } from "dotenv";
config();

global.indexedDB = fakeIndexedDB;
global.IDBKeyRange = fakeIndexedDB.IDBKeyRange;

var localStorage = new LocalStorage("./scratch");

const db = new Datastore({ filename: "path/to/datafile", autoload: true });

export const run = async (query, filename, chat_History) => {
  let loader = null;
  let docs = null;
  // console.log(privateKey,url)
  if (filename !== null) {
    try {
      loader = new PDFLoader("book.pdf");
      docs = await loader.load();

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });
      const docOutput = await splitter.splitDocuments(docs);

      const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_KEY,
      });

      let vectorStore = await MemoryVectorStore.fromDocuments(
        docOutput,
        embeddings
      );

      const chatHistory = JSON.parse(chat_History);

      const formattedChatHistory = chatHistory.map((message) => {
        if (message?.user === "user") {
          return new HumanMessage(message.message);
        } else {
          return new AIMessage(message.message);
        }
      });
      // console.log(formattedChatHistory);

      const memory = new BufferMemory({
        memoryKey: "chat_history",
        returnMessages: true,
        chatHistory: new ChatMessageHistory(formattedChatHistory),
      });

      const model = new ChatOpenAI({
        openAIApiKey: process.env.OPENAI_KEY,
        modelName: "gpt-3.5-turbo",
      });
      const chain = ConversationalRetrievalQAChain.fromLLM(
        model,
        vectorStore.asRetriever(),
        {
          memory,
        }
      );

      const result = await chain.call({
        question: query,
      });
      return result;
    } catch (err) {
      console.log(err);
    }
  }
};