from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.conversational import prompt
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

load_dotenv(".env")


def main():
    documents = SimpleDirectoryReader("data").load_data()
    index = GPTSimpleVectorIndex.from_documents(
        documents,
    )

    tools = [
        Tool(
            name="Kanmon Tunnel",
            func=lambda q: str(index.query(q)),
            description="田中太郎の夕飯の献立を調べる際に利用することができます。",
            return_direct=True,
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=OpenAI(temperature=0),
        agent="conversational-react-description",
        verbose=True,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        agent_kwargs=dict(suffix="Answer should be in Japanese.¥n" + prompt.SUFFIX),
    )

    print(agent.agent.llm_chain.prompt.template)
    while True:
        user_input = input("input:")
        if user_input == "exit":
            break
        print(agent.run(user_input))


if __name__ == "__main__":
    main()
