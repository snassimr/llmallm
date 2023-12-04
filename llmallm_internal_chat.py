
#################################################################################
from dotenv import load_dotenv
# Load .env file
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentType

def get_random_response(q):
    import random
    response = random.choice(["How are you?", "I love you", "I'm very hungry"])
    return response

get_response = get_random_response

tools = [
    Tool(
        name="LlamaQueryEngine",
        func=lambda q: get_response(q),
        description="Useful for answering any question.",
        return_direct=True,
    ),
]

# set Logging to DEBUG for more detailed outputs


memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

llm = ChatOpenAI(temperature=0)
agent_executor = initialize_agent(
    tools, llm, agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    memory=memory, 
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs=agent_kwargs, 
)

while True:
    text_input = input("User : ")
    print("User : " + text_input)
    if (text_input.lower() == 'stop'):
        break
    response= agent_executor.run(input=text_input)

    print(f'Agent: {response}')

