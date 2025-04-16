# Importing modules
import asyncio
import os
import sys
from typing import Annotated, Union, AsyncGenerator
from autogen_agentchat.messages import ChatMessage

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult
from mmct.video_pipeline.tools.get_summary_n_transcript import (
    get_summary_n_transcript,
)
from mmct.video_pipeline.tools.query_gpt4_vision import query_GPT4_Vision
from mmct.video_pipeline.tools.query_summary_n_transcript import (
    query_summary_n_transcript,
)
from mmct.video_pipeline.tools.query_frame_azure_computer_vision import (
    query_frames_Azure_Computer_Vision,
)
from mmct.video_pipeline.tools.critic import critic_tool
from mmct.video_pipeline.prompts_and_description import (
    SYSTEM_PROMPT_PLANNER_WITH_GUARDRAILS,
    SYSTEM_PROMPT_PLANNER_WITHOUT_CRITIC,
    CRITIC_AGENT_SYSTEM_PROMPT,
    PLANNER_DESCRIPTION,
    CRITIC_DESCRIPTION,
)
from mmct.video_pipeline.utilities.helper import load_required_files
from mmct.llm_client import LLMClient
from dotenv import load_dotenv

load_dotenv(override=True)

# Setting global encoding to utf-8
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")


class VideoQnA:
    """
    videoQnA handles video-based queries using MMCT's agent-based approach. It uses various tools to extract
    transcripts, summaries, and visual frame-level insights from videos. The pipeline supports a planner and
    an optional critic agent to ensure high-quality, validated responses.

    Parameters:
        query (str): Question or prompt related to the video.
        video_id (str): video reference ID to query.
        critic_flag (bool): Whether to include a critic agent for response validation.
        tools_str (str): Comma-separated tool names to use (e.g., 'get_summary_transcript,query_GPT4_Vision').

    Example Usage:
    --------------

    Non-Streaming Response:
    -----------------------
    >>> from mmct.video_pipeline.agents.video_qna import videoQnA
    >>> import asyncio
    >>> async def run_non_stream():
    >>>     video_qna = videoQnA(
    >>>         video_id="bcFvbtZafKM",
    >>>         query="What is IPL 321?",
    >>>         critic_flag=True,
    >>>         tools_str="get_summary_transcript,query_summary_transcript"
    >>>     )
    >>>     result = await video_qna.run()
    >>>     print(result)
    >>> asyncio.run(run_non_stream())

    Streaming Response:
    -------------------
    >>> from mmct.video_pipeline.agents.video_qna import videoQnA
    >>> from autogen_agentchat.base import TaskResult
    >>> import asyncio
    >>> async def run_stream():
    >>>     video_qna = videoQnA(
    >>>         video_id="bcFvbtZafKM",
    >>>         query="What is IPL 321?",
    >>>         critic_flag=True,
    >>>         tools_str="get_summary_transcript,query_GPT4_Vision",
    >>>     )
    >>>     stream_response = await video_qna.run_stream()
    >>>     async for response in stream_response:
    >>>         if not isinstance(response, TaskResult):
    >>>             print("\\n", response.content, flush=True)
    >>> asyncio.run(run_stream())
    """

    def __init__(
        self,
        query,
        video_id,
        critic_flag=True,
        tools_str="get_summary_transcript,query_GPT4_Vision,query_summary_transcript,query_frames_Azure_Computer_Vision",
    ):
        self.query = query
        self.video_id = video_id
        self.tools_str = tools_str
        self.critic_flag = critic_flag

        service_provider = os.getenv("LLM_PROVIDER", "azure")
        self.model_client = LLMClient(
            autogen=True, service_provider=service_provider
        ).get_client()

        self.tools_list = []
        self.planner_agent = None
        self.critic_agent = None
        self.team = None

        self.task = f"query:{self.query}.,\nInstruction:video id:{self.video_id}"

    def _initialize_tools(self):
        tool_map = {
            "get_summary_transcript": get_summary_n_transcript,
            "query_summary_transcript": query_summary_n_transcript,
            "query_GPT4_Vision": query_GPT4_Vision,
            "query_frames_Azure_Computer_Vision": query_frames_Azure_Computer_Vision,
        }
        self.tools = [tool_map[t] for t in self.tools_str.split(",") if t in tool_map]

    async def _initialize_agents(self):
        # system prompt for video planner agent
        planner_system_prompt = (
            SYSTEM_PROMPT_PLANNER_WITH_GUARDRAILS
            if self.critic_flag
            else SYSTEM_PROMPT_PLANNER_WITHOUT_CRITIC
        )

        # Define Planner agent
        self.planner = AssistantAgent(
            name="planner",
            model_client=self.model_client,
            model_client_stream=False,
            description=PLANNER_DESCRIPTION,
            system_message=(f"""{planner_system_prompt}"""),
            tools=self.tools,
            reflect_on_tool_use=True,
        )

        if self.critic_flag:
            self.critic = AssistantAgent(
                name="critic",
                model_client=self.model_client,
                model_client_stream=True,
                description=CRITIC_DESCRIPTION,
                system_message=(f"{CRITIC_AGENT_SYSTEM_PROMPT}"),
                tools=[critic_tool],
                reflect_on_tool_use=False,
            )

            text_mention_termination = TextMentionTermination("TERMINATE")
            max_messages_termination = MaxMessageTermination(max_messages=250)
            termination = text_mention_termination | max_messages_termination

            selector_prompt = """Select an agent to perform task.

            {roles}

            Current conversation context:
            {history}

            Read the above conversation, then select an agent from {participants} to perform the next task.
            Make sure 'critic' agent comes only when planner ask for criticism or feedback.
            For your information - There are only two agents - 'planner' & 'critic'
            Only select one agent.
            """
            self.team = SelectorGroupChat(
                [self.planner, self.critic],
                model_client=self.model_client,
                termination_condition=termination,
                allow_repeated_speaker=True,
                selector_prompt=selector_prompt,
            )

    async def setup(self):
        self._initialize_tools()
        await self._initialize_agents()

    async def run(self):
        await self.setup()

        if self.critic_flag:
            result = await self.team.run(task=self.task)
        else:
            result = await self.planner.run(task=self.task)

        return result.messages[-1]

    async def run_stream(self):
        await self.setup()

        if self.critic_flag:
            return self.team.run_stream(task=self.task)
        else:
            return self.planner.run_stream(task=self.task)


async def video_qna(
    query: Annotated[str, "The question to be answered based on the content of the video."],
    video_id: Annotated[str, "The unique identifier of the video."],
    critic_flag: Annotated[bool, "Set to True to enable a critic agent that validates the response."] = True,
    tools_str: Annotated[str, "Comma-separated list of tools to use when answering the query."] = "get_summary_transcript,query_GPT4_Vision,query_summary_transcript,query_frames_Azure_Computer_Vision",
    stream: Annotated[bool, "Set to True to return the response as a stream."] = False,
) -> Union[ChatMessage, AsyncGenerator[ChatMessage, None]]:
    """
    Answers a user query based on the content of a specified video.

    This tool combines various analysis components (e.g., transcript summarization, visual understanding)
    to generate accurate responses. It supports optional response validation via a critic agent
    and can operate in both standard and streaming modes.
    """
    await load_required_files(session_id=video_id)
    video_qna_instance = VideoQnA(
        video_id=video_id,
        query=query,
        critic_flag=critic_flag,
        tools_str=tools_str,
    )

    if stream:
        return await video_qna_instance.run_stream()
    else:
        return await video_qna_instance.run()




if __name__ == "__main__":
    video_id = "bcFvbtZafKM"
    query = "What is IPL 321?"
    stream = True
    critic_flag = True
    tools_str = "get_summary_transcript,query_GPT4_Vision,query_summary_transcript,query_frames_Azure_Computer_Vision"

    if stream:
        async def run_stream():
            stream_response = await video_qna(
                query=query,
                video_id=video_id,
                critic_flag=critic_flag,
                tools_str=tools_str,
                stream=True,
            )
            async for response in stream_response:
                if not isinstance(response, TaskResult):
                    print(response.content, end="", flush=True)

        asyncio.run(run_stream())

    else:
        async def run_non_stream():
            response = await video_qna(
                query=query,
                video_id=video_id,
                critic_flag=critic_flag,
                tools_str=tools_str,
                stream=False,
            )
            print(response.content)

        asyncio.run(run_non_stream())


