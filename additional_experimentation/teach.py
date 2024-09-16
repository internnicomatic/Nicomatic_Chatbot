from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen import ConversableAgent 
from autogen import UserProxyAgent
## configure Ollama wrapper with default portnumber 
config_list = [
  {
    "model": "llama3.1:8b",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama"}]
# configure the LLM
llm_config={"config_list": config_list}
## Create an Agent 
teachable_agent = ConversableAgent(
    name="teachable_agent",  
    llm_config=llm_config,
    code_execution_config={"use_docker": False})
## Configure the teachability function  
teachability = Teachability(
    reset_db=False,  
    path_to_db_dir="./tmp/interactive/teachability_db")
## incorporate this teachability into the Agent
teachability.add_to_agent(teachable_agent)
## Define use input function
user = UserProxyAgent("user", human_input_mode="ALWAYS",  code_execution_config={"use_docker": False})
## Infer
teachable_agent.initiate_chat(user, message="Hi, I'm a teachable user assistant! What's on your mind?")