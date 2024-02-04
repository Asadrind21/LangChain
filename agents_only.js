import { config } from "dotenv";
config();

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createRetrieverTool } from "langchain/tools/retriever";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { createOpenAIFunctionsAgent } from "langchain/agents";
import { AgentExecutor } from "langchain/agents";

const searchTool = new TavilySearchResults();


const loader = new CheerioWebBaseLoader(
    "https://docs.smith.langchain.com/overview"
  );
  const rawDocs = await loader.load();
  
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await splitter.splitDocuments(rawDocs);
  
  const vectorstore = await MemoryVectorStore.fromDocuments(
    docs,
    new OpenAIEmbeddings()
  );
  const retriever = vectorstore.asRetriever();
  
  const retrieverTool = createRetrieverTool(retriever, {
    name: "langsmith_search",
    description:
      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
  });

  const tools = [searchTool, retrieverTool];

  const llm = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0,
  });

// Get the prompt to use - you can modify this!
// If you want to see the prompt in full, you can at:
// https://smith.langchain.com/hub/hwchase17/openai-functions-agent
const prompt = await pull<ChatPromptTemplate>(
    "hwchase17/openai-functions-agent"
  );

  const agent = await createOpenAIFunctionsAgent({
    llm,
    tools,
    prompt,
  });
  const agentExecutor = new AgentExecutor({
    agent,
    tools,
  });

  const result1 = await agentExecutor.invoke({
  input: "hi!",
});

console.log(result1);