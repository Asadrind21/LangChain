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

import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { createRetrieverTool } from "langchain/tools/retriever";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { pull } from "langchain/hub";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";



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



const retriever = vectorstore.asRetriever();


  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);
  
  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever,
    rephrasePrompt: historyAwarePrompt,
  });

  const chatHistory = [
    new HumanMessage("Can Vitamin A increase immunity?"),
    new AIMessage("Yes!"),
  ];
  
  await historyAwareRetrieverChain.invoke({
    chat_history: chatHistory,
    input: "how can i get it!",
  });

  const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions based on the below context:\n\n{context}",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);
  
  const historyAwareCombineDocsChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: historyAwareRetrievalPrompt,
  });
  
  const conversationalRetrievalChain = await createRetrievalChain({
    retriever: historyAwareRetrieverChain,
    combineDocsChain: historyAwareCombineDocsChain,
  });

  const result2 = await conversationalRetrievalChain.invoke({
    chat_history: [
      new HumanMessage("Can Vitamic C help boost immune system?"),
      new AIMessage("Yes!"),
    ],
    input: "tell me how",
  });
  
  //console.log(result2.answer);
  ////////

const retrieverTool = await createRetrieverTool(retriever, {
name: "disease_search",
description:
    "Search for information about disease and their diagnosis. For any questions about disease diagnosis and prediction, you must use this tool!",
});


const searchTool = new TavilySearchResults();
const tools = [retrieverTool, searchTool];

// Get the prompt to use - you can modify this!
// If you want to see the prompt in full, you can at:
// https://smith.langchain.com/hub/hwchase17/openai-functions-agent
const agentPrompt = await pull<ChatPromptTemplate>(
    "hwchase17/openai-functions-agent"
  );
  
  const agentModel = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106",
    temperature: 0,
  });
  
  const agent = await createOpenAIFunctionsAgent({
    llm: agentModel,
    tools,
    prompt: agentPrompt,
  });
  
  const agentExecutor = new AgentExecutor({
    agent,
    tools,
    verbose: true,
  });

  const agentResult = await agentExecutor.invoke({
    input: "how can LangSmith help with testing?",
  });
  
  console.log(agentResult.output);