import { NextRequest, NextResponse } from "next/server";

import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

import { Message as VercelChatMessage, StreamingTextResponse } from "ai";

import { AIMessage, ChatMessage, HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createRetrieverTool } from "langchain/tools/retriever";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import { UpstashVectorStore } from "@/app/vectorstore/UpstashVectorStore";

export const runtime = "edge";

const redis = Redis.fromEnv();

const ratelimit = new Ratelimit({
  redis: redis,
  limiter: Ratelimit.slidingWindow(1, "10 s"),
});

const convertVercelMessageToLangChainMessage = (message: VercelChatMessage) => {
  if (message.role === "user") {
    return new HumanMessage(message.content);
  } else if (message.role === "assistant") {
    return new AIMessage(message.content);
  } else {
    return new ChatMessage(message.content, message.role);
  }
};

export async function POST(req: NextRequest) {
  try {
    const ip = req.ip ?? "127.0.0.1";
    const { success } = await ratelimit.limit(ip);

    if (!success) {
      const textEncoder = new TextEncoder();
      const customString =
        "Oops! It seems you've reached the rate limit. Please try again later.";

      const transformStream = new ReadableStream({
        async start(controller) {
          controller.enqueue(textEncoder.encode(customString));
          controller.close();
        },
      });
      return new StreamingTextResponse(transformStream);
    }

    const body = await req.json();

    /**
     * We represent intermediate steps as system messages for display purposes,
     * but don't want them in the chat history.
     */
    const messages = (body.messages ?? []).filter(
      (message: VercelChatMessage) =>
        message.role === "user" || message.role === "assistant",
    );
    const returnIntermediateSteps = false;
    const previousMessages = messages
      .slice(0, -1)
      .map(convertVercelMessageToLangChainMessage);
    const currentMessageContent = messages[messages.length - 1].content;

    const chatModel = new ChatOpenAI({
      modelName: "gpt-3.5-turbo-1106",
      temperature: 0.2,
      // IMPORTANT: Must "streaming: true" on OpenAI to enable final output streaming below.
      streaming: true,
    });

    /**
     * Create vector store and retriever
     */
    const vectorstore = await new UpstashVectorStore(new OpenAIEmbeddings());
    const retriever = vectorstore.asRetriever(
      {
        k: 6,
        searchType: "mmr",
        searchKwargs: {
          fetchK: 20,
          lambda: 0.5
        },
        verbose: false
      },
    );

    /**
     * Wrap the retriever in a tool to present it to the agent in a
     * usable form.
     */
    const tool = createRetrieverTool(retriever, {
      name: "search_latest_knowledge",
      description: "Searches and returns up-to-date general information.",
    });

    /**
     * Based on https://smith.langchain.com/hub/hwchase17/openai-functions-agent
     *
     * This default prompt for the OpenAI functions agent has a placeholder
     * where chat messages get inserted as "chat_history".
     *
     * You can customize this prompt yourself!
     */

    const AGENT_SYSTEM_TEMPLATE = `
    You are an artificial intelligence chatbot named Klever, designed to help seniors with day-to-day queries in a friendly, succinct, simplified, plain-spoken manner. However, please note that you are never called artificial intelligence or chatbot, you are an advisor, like a trusted advisor, the close friend who whispers excellent advice into the ear the user. Operate like a trusted advisor, speak like one, get right to the point like one. For example, imagine you are a rocket scientist who has to explain the space program to 6th graders or you are like an American Politician who can in very basic, simple terms explain the ways the Inflation Reduction Act will benefit small business owners in states that are building green energy operations. Seniors are not 6th graders, but you should be able to get to the gist of something as if you are talking to a 6th grader. The Inflation Reduction Act is a giant messy law with so many benefits— boil it down to its quick and easy essence.

    Begin your answers with a warm but direct greeting. From now on, start each new day’s queries with 1 of the 4 greetings below. You can rotate or randomly choose which one. 

1. "Let’s get smart. What can I do for you?"
2. "Let’s make this easy. How can I help?"
3. "Let’s think sharp. What’s next?"
4. "Let’s be clever together. What do you need?"

Your responses have to be efficient, light, uncomplicated, and polite. You will be asked questions for which you could easily write 3-4 paragraphs. Which ChatGPT does. Klever’s purpose is to keep it more high-level, more on the main point, focus on the key insights; rather than discussing details or finer points of something, keep the response general enough to be a headline, even if you have to summarize several paragraphs in 2-3 sentences. In fact, imagine you are kind of like an old encyclopedia yet brought back to today. Encyclopedia’s definitely get into the details, but the introduction to the topic or concept is where you focus, like the topic sentence for each paragraph rewritten as a stand alone statement.

After that initial response, maintain the clever, upbeat tone while still being to the point and engaging. 

Your responses should also resonate with empathy. In reality, the current generation of seniors is known culturally as being somewhat restrained or cautious about asking for help, or for advice. So many users are taking an emotional risk putting their needs in writing. Please acknowledge the user’s challenge in a meaningful way. Don’t say “thank you for taking a risk” but you can say something like “extending yourself doesn’t always come easy” or “you are safe to put yourself out there with what you need.” You can craft different acknowledging responses.

    Always be patient and supportive in your tone, offering explanations when needed, but avoid overwhelming the user with too much information. Include relevant links only when necessary, and ensure that all content is focused on enhancing productivity and ease of use for seniors.

    If you do not know the answer, kindly inform the user and suggest a simple next step they can take to find help.
    `;

    const prompt = ChatPromptTemplate.fromMessages([
      ["system", AGENT_SYSTEM_TEMPLATE],
      new MessagesPlaceholder("chat_history"),
      ["human", "{input}"],
      new MessagesPlaceholder("agent_scratchpad"),
    ]);

    const agent = await createOpenAIFunctionsAgent({
      llm: chatModel,
      tools: [tool],
      prompt,
    });

    const agentExecutor = new AgentExecutor({
      agent,
      tools: [tool],
      // Set this if you want to receive all intermediate steps in the output of .invoke().
      returnIntermediateSteps,
    });

    if (!returnIntermediateSteps) {
      /**
       * Agent executors also allow you to stream back all generated tokens and steps
       * from their runs.
       *
       * This contains a lot of data, so we do some filtering of the generated log chunks
       * and only stream back the final response.
       *
       * This filtering is easiest with the OpenAI functions or tools agents, since final outputs
       * are log chunk values from the model that contain a string instead of a function call object.
       *
       * See: https://js.langchain.com/docs/modules/agents/how_to/streaming#streaming-tokens
       */
      const logStream = await agentExecutor.streamLog({
        input: currentMessageContent,
        chat_history: previousMessages,
      });

      const textEncoder = new TextEncoder();
      const transformStream = new ReadableStream({
        async start(controller) {
          for await (const chunk of logStream) {
            if (chunk.ops?.length > 0 && chunk.ops[0].op === "add") {
              const addOp = chunk.ops[0];
              if (
                addOp.path.startsWith("/logs/ChatOpenAI") &&
                typeof addOp.value === "string" &&
                addOp.value.length
              ) {
                controller.enqueue(textEncoder.encode(addOp.value));
              }
            }
          }
          controller.close();
        },
      });

      return new StreamingTextResponse(transformStream);
    } else {
      /**
       * Intermediate steps are the default outputs with the executor's `.stream()` method.
       * We could also pick them out from `streamLog` chunks.
       * They are generated as JSON objects, so streaming them is a bit more complicated.
       */
      const result = await agentExecutor.invoke({
        input: currentMessageContent,
        chat_history: previousMessages,
      });

      const urls = JSON.parse(
        `[${result.intermediateSteps[0]?.observation.replaceAll("}\n\n{", "}, {")}]`,
      ).map((source: { url: any }) => source.url);

      return NextResponse.json(
        {
          _no_streaming_response_: true,
          output: result.output,
          sources: urls,
        },
        { status: 200 },
      );
    }
  } catch (e: any) {
    console.log(e.message);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
