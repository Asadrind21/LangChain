import { config } from "dotenv";
config();

import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const apiKey = process.env.OPENAI_API_KEY;

const chatModel = new ChatOpenAI({apiKey});

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

// console.log(docOutput.length);
// console.log(docOutput[0].pageContent.length);
//  console.log(docOutput);

const embeddings = new OpenAIEmbeddings();

const vectorstore = await MemoryVectorStore.fromDocuments(
    docOutput,
    embeddings
);


const systemTemplate =
  "You are a medical praticionar that helps patients diagnose their problems based on the {context} and give them suitable recommendations.";
const humanTemplate = "{input}";

const chatPrompt = ChatPromptTemplate.fromMessages([
  ["system", systemTemplate],
  ["human", humanTemplate],
]);

const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: chatPrompt,
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