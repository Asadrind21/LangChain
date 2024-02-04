import { config } from "dotenv";
config();

import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";
import { PineconeTranslator } from "langchain/retrievers/self_query/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
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
      new Document({ pageContent: String(book) }),
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
    
// if (
//     !process.env.PINECONE_API_KEY ||
//     !process.env.PINECONE_ENVIRONMENT ||
//     !process.env.PINECONE_INDEX
//   ) {
//     throw new Error(
//       "PINECONE_ENVIRONMENT and PINECONE_API_KEY and PINECONE_INDEX must be set"
//     );
//   }
  
//   const pinecone = new Pinecone();
  
//   const index = pinecone.Index(process.env.PINECONE_INDEX);

//   const embeddings = new OpenAIEmbeddings();
//   const llm = new OpenAI();
//   const documentContents = "Book about medical diagnosis";
//   const vectorStore = await PineconeStore.fromDocuments(docs, embeddings, {
//     pineconeIndex: index,
//   });
//   const selfQueryRetriever = await SelfQueryRetriever.fromLLM({
//     llm,
//     vectorStore,
//     documentContents,
//     /**
//      * We need to create a basic translator that translates the queries into a
//      * filter format that the vector store can understand. We provide a basic translator
//      * translator here, but you can create your own translator by extending BaseTranslator
//      * abstract class. Note that the vector store needs to support filtering on the metadata
//      * attributes you want to query on.
//      */
//     structuredQueryTranslator: new PineconeTranslator(),
//   });
  
//   /**
//    * Now we can query the vector store.
//    * We can ask questions like "Which movies are less than 90 minutes?" or "Which movies are rated higher than 8.5?".
//    * We can also ask questions like "Which movies are either comedy or drama and are less than 90 minutes?".
//    * The retriever will automatically convert these questions into queries that can be used to retrieve documents.
//    */
//   const query1 = await selfQueryRetriever.getRelevantDocuments(
//     "Which treatments are for cardiac arrest?"
//   );
//   const query2 = await selfQueryRetriever.getRelevantDocuments(
//     "What in the normal blood pressure?"
//   );
//   const query3 = await selfQueryRetriever.getRelevantDocuments(
//     "i have chilblain, what should i do?"
//   );
 
//   console.log(query1, query2, query3);
