# Standard Library
import asyncio
from typing import Optional
from dotenv import load_dotenv

# Local Imports
from mmct.video_pipeline.core.tools.video_qna import video_qna
from mmct.video_pipeline.prompts_and_description import (
    VIDEO_AGENT_SYSTEM_PROMPT,
    VideoAgentResponse,
)
from mmct.config.settings import MMCTConfig
from mmct.providers.factory import provider_factory
from mmct.utils.logging_config import LoggingConfig

# Load environment variables
load_dotenv(override=True)


class VideoAgent:
    """
    Simplified agent for video question answering using the updated video_qna function.

    This agent provides a clean interface that:
    1. Calls video_qna with the provided parameters
    2. Formats the response using LLM with structured output
    3. Returns a properly structured VideoAgentResponse

    Args:
        query (str): The natural language question about video content.
        index_name (str): Name of the Azure Cognitive Search index for video retrieval.
        video_id (Optional[str]): Specific video ID to query. Defaults to None.
        youtube_url (Optional[str]): YouTube URL to query. Defaults to None.
        use_critic_agent (bool): Whether to use the critic agent for validation. Defaults to True.
        stream (bool): Whether to stream the response output. Defaults to False.
        llm_provider (Optional[object]): LLM provider instance. Defaults to None (uses config).
    """

    def __init__(
        self,
        query: str,
        index_name: str,
        video_id: Optional[str] = None,
        youtube_url: Optional[str] = None,
        use_critic_agent: bool = True,
        stream: bool = False,
        llm_provider: Optional[object] = None
    ):
        # Store parameters
        self.query = query
        self.index_name = index_name
        self.video_id = video_id
        self.youtube_url = youtube_url
        self.use_critic_agent = use_critic_agent
        self.stream = stream

        # Initialize configuration and logging
        self.config = MMCTConfig()
        self._setup_logging()

        # Initialize LLM provider
        self.llm_provider = llm_provider or self._create_llm_provider()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        LoggingConfig.setup_logging(
            level=self.config.logging.level,
            log_file=self.config.logging.log_file if self.config.logging.enable_file_logging else None,
            enable_json=self.config.logging.enable_json
        )

    def _create_llm_provider(self) -> object:
        """Create LLM provider from configuration."""
        return provider_factory.create_llm_provider(
            self.config.llm.provider,
            self.config.llm.model_dump()
        )

    async def __call__(self) -> VideoAgentResponse:
        """
        Main execution method for the VideoAgent.

        Returns:
            VideoAgentResponse: Structured response containing the answer to the query.
        """
        try:
            # Call the video_qna function with simplified parameters
            # Get response from video_qna
            video_qna_response = await video_qna(
                query=self.query,
                video_id=self.video_id,
                youtube_url=self.youtube_url,
                use_critic_agent=self.use_critic_agent,
                index_name=self.index_name,
                stream=self.stream,
                llm_provider=self.llm_provider
            )

            # Generate final formatted answer using LLM with video_qna response
            formatted_response = await self._generate_final_answer(video_qna_response)
            return formatted_response

        except Exception as e:
            return self._create_error_response(f"VideoAgent execution failed: {str(e)}")

    async def _generate_final_answer(self, video_qna_response: dict) -> VideoAgentResponse:
        """
        Use LLM to generate a final consolidated and structured answer.

        Args:
            video_qna_response: Response from video_qna function

        Returns:
            VideoAgentResponse: Formatted response using pydantic model
        """
        try:
            # Prepare context and messages
            context_text = str(video_qna_response)
            messages = self._prepare_messages(context_text)

            # Get structured response from LLM
            response = await self.llm_provider.chat_completion(
                messages=messages,
                temperature=self.config.llm.temperature,
                response_format=VideoAgentResponse
            )
            return response

        except Exception as e:
            return self._create_error_response(f"Error generating final answer: {str(e)}")

    def _prepare_messages(self, context_text: str) -> list:
        """Prepare messages for LLM completion."""
        return [
            {"role": "system", "content": VIDEO_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query: {self.query}\nContext: {context_text}"
            }
        ]

    def _create_error_response(self, error_message: str) -> VideoAgentResponse:
        """Create a structured error response."""
        return VideoAgentResponse(
            response=error_message,
            answer_found=False,
            source=[],
            tokens={"input_token": 0, "output_token": 0}
        )


if __name__ == "__main__":

    async def main():
        """Example usage of VideoAgent."""
        query = "What is the tutor wearing in the video?"
        youtube_url = "https://youtube.com/watch?v=U1JYwHcFfso"
        index_name = "education-video-index-v2"

        video_agent = VideoAgent(
            query=query,
            youtube_url=youtube_url,
            index_name=index_name,
            use_critic_agent=True,
            stream=False
        )

        results = await video_agent()
        print("-" * 60)
        print(f"Query: {query}")
        print("-" * 60)
        print(results)

    asyncio.run(main())