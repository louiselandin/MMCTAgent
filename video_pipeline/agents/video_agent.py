# Importing modules
import asyncio
import os
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
from mmct.video_pipeline.tools.video_qna import video_qna
from mmct.video_pipeline.tools.video_search import video_search
from mmct.video_pipeline.prompts_and_description import VIDEO_AGENT_SYSTEM_PROMPT

from mmct.llm_client import LLMClient
from dotenv import load_dotenv

load_dotenv(override=True)
# Setting global encoding to utf-8
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

class VideoAgent:
    def __init__(self, query, top_n=None):
        self.query = query
        self.tools = [video_qna, video_search] # Ingestion tool
        if top_n is not None:
            self.top_n=top_n
        service_provider = os.getenv("LLM_PROVIDER", "azure")
        self.model_client = LLMClient(
            autogen=True, service_provider=service_provider
        ).get_client()
        
    async def _initialize_agents(self):
        self.video_agent = AssistantAgent(
            name="video_agent",
            model_client=self.model_client,
            model_client_stream=False,
            description="An intelligent video analysis agent that answers user queries by searching for relevant videos and extracting accurate information using video content understanding tools.",
            system_message=f"{VIDEO_AGENT_SYSTEM_PROMPT}",
            tools=self.tools,
            reflect_on_tool_use=False,
            
        )
        
    async def setup(self):
        await self._initialize_agents()
        
    async def run_stream(self):
        await self.setup()
        
        #Termination Condition
        text_mention_termination = TextMentionTermination("TERMINATE")
        max_messages_termination = MaxMessageTermination(max_messages=20)
        termination = text_mention_termination | max_messages_termination
        
        self.team = RoundRobinGroupChat(participants=[self.video_agent], termination_condition=termination)
        self.task = f"query:{self.query}"
        if self.top_n is not None:
            self.task += f"\n top_n:{self.top_n} retrevel of video ids"
        result = self.team.run_stream(task=self.task)
        return result
    
if __name__ == "__main__":
    query = "How is Beejamrut prepared, and what are its benefits?"
    top_n = 3
    #video_id = "bcFvbtZafKM"

    async def main():
        video_agent = VideoAgent(query=query, top_n=top_n)
        stream_response = await video_agent.run_stream()
        async for response in stream_response:
            if not isinstance(response, TaskResult):
                print("\n", response, flush=True)
            # else:
            #     print("\nTask completed:", response, flush=True)

    asyncio.run(main())
