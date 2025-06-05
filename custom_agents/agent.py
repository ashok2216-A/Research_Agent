from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from custom_agents.tools import youtube_search, search_datasets, academic_search

def create_agent(api_key):
    if not api_key:
        return None
    try:
        llm = ChatMistralAI(model="mistral-small-latest", temperature=0.1, mistral_api_key=api_key)

        memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True,output_key="output")
        
        tools = [Tool(name="academic_search",
                func=academic_search,
                description="Search for academic papers and research publications. Provide a research topic or keywords."),
            Tool(name="dataset_search",
                func=search_datasets,
                description="Search for research datasets. Provide a topic or keywords to find relevant datasets."),
            Tool(name="youtube_search",
                func=youtube_search,
                description="Search for YouTube videos. Use this when you need to find videos on a specific topic.")]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,  # Add memory
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        return agent
    except Exception as e:
        raise Exception(f"Error initializing agent: {str(e)}")

def get_agent_response(agent, prompt):
    try:
        response = agent.run(input=prompt)
        return response
    except Exception as e:
        return f"Error getting agent response: {str(e)}"

def clear_agent_memory(agent):
    try:
        agent.memory.clear()
        return "Memory cleared successfully"
    except Exception as e:
        return f"Error clearing memory: {str(e)}"
