# Importing modules
from openai import AzureOpenAI
from openai import ContentFilterFinishReasonError, RateLimitError, APITimeoutError
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from mmct.llm_client import LLMClient
from typing_extensions import Annotated
import os
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

openai_client = LLMClient().get_client()


class VideoSearch:
    def __init__(
        self, query, index_name="jharkhand-video-index-v2", top_n=3, min_threshold=80
    ):
        self.query = query
        self.top_n = top_n
        self.index_name = index_name
        self.min_threshold = min_threshold

    async def generate_embeddings(self, text: str, openai_client):
        """Function to generate embeddings for the given text

        Args:
            text (str): input string


        Returns:
            [list]: OpenAI Embeddings
        """
        try:
            response = await asyncio.to_thread(
                openai_client.embeddings.create,
                input=[text],
                model=os.environ.get("EMBEDDING_DEPLOYMENT"),
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Exception occured while creating embeddings: {e}")

    async def Species_and_variety_query(self, transcript, openai_client):
        try:
            json_format = {
                "species": "Name of the species which is talked in the video",
                "Variety_of_species": "Name the variety of species which is mention in the video",
            }
            system_prompt = f"""
                You are a VideoAnalyzerGPT. Your job is to find all the details from the video frames of every 2 seconds and from the audio.
                Mention only the english name or the text into the response, if the text is mention in the video is in hindi or any language then convert them into english language.
                If any text from anywhere video frames or transcript is in hindi then translate them into english and then include it into response.
                Topics that you have to find and given in the response:
                1. Species name which is talked in the video.
                2. Specific Variety of species(e.g. IPA 15-06, IPL 203, IPH 15 03 etc) on which they are talking.
                If transcript does not contains any species or variety of species then assign 'None'.
                Make sure include response languge is only english. not hinglish or hindi etc.
                Make sure to add the english translated name of species and their variety.
                Only when sure then only add the name of species or variety of species.
                Provide the final response into the below given json: Json format: {json_format}
                    Note: Dont provide ```json in the response.
                """
            prompt = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The audio transcription is: {transcript}",
                        }
                    ],
                },
            ]
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model=os.environ.get("AZURE_OPENAI_MODEL"),
                messages=prompt,
                temperature=0,
            )
            return response.choices[0].message.content
        except ContentFilterFinishReasonError as e:
            raise Exception(f"Content Filtering Error raised from OpenAI: {e}")
        except RateLimitError as e:
            raise Exception(f"Rate Limit Error occured from openAI: {e}")
        except APITimeoutError as e:
            raise Exception(f"API Timeout Error from OpenAI: {e}")
        except Exception as e:
            if "OpenAI Bad Request Error 400" in str(
                e
            ) or "response was filtered due to the prompt triggering" in str(e):
                return "{'species': 'None', 'Variety_of_species': 'None'}"
            else:
                return f"error: {e}"

    async def search_ai(
        self,
        query,
        index_name,
        top_n,
        min_threshold,
        openai_client,
        species=None,
        variety=None,
    ):
        try:
            min_threshold = min_threshold / 100
            # setting up the environment variables and required azure clients
            AZURE_MANAGED_IDENTITY = os.environ.get(
                "AZURE_OPENAI_MANAGED_IDENTITY", None
            )
            azure_search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT", None)
            if azure_search_endpoint is None:
                raise Exception("Azure search endpoint is missing in env!")

            if AZURE_MANAGED_IDENTITY is None:
                raise Exception(
                    "AZURE_OPENAI_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )

            if AZURE_MANAGED_IDENTITY.upper() == "TRUE":
                index_client = SearchClient(
                    endpoint=azure_search_endpoint,
                    index_name=index_name,
                    credential=DefaultAzureCredential(),
                )
            else:
                azure_search_key = os.environ.get("AZURE_SEARCH_KEY", None)
                if azure_search_key is None:
                    raise Exception("Azure Search Key is missing!")
                index_client = SearchClient(
                    endpoint=azure_search_endpoint,
                    index_name=index_name,
                    credential=AzureKeyCredential(key=azure_search_key),
                )
            query_embds = await self.generate_embeddings(
                text=query, openai_client=openai_client
            )
            vector_query = VectorizedQuery(vector=query_embds, fields="embeddings")

            filter_expression = []

            if species:
                filter_expression.append(f"search.ismatch('\"{species}\"', 'species')")

            if variety:
                filter_expression.append(f"search.ismatch('\"{variety}\"', 'variety')")

            if filter_expression:
                filter_query = " and ".join(filter_expression)
            else:
                filter_query = None
            results = await index_client.search(
                search_text=None,
                vector_queries=[vector_query],
                vector_filter_mode=VectorFilterMode.PRE_FILTER,
                top=50,
                filter=filter_query,
                select=["content", "species", "variety", "url", "url_id"],
            )
            filtered_results = []
            async for result in results:
                if min_threshold <= result["@search.score"]:
                    filtered_results.append(result)

            top_n_results = []
            seen_urls = set()
            for result in filtered_results:
                if result["url"] not in seen_urls:
                    seen_urls.add(result["url"])
                    top_n_results.append(result)
                if len(top_n_results) == top_n:
                    break
            return top_n_results
        except Exception as e:
            raise Exception(f"Error while doing AI search: {e}")
        finally:
            if index_client:
                await index_client.close()

    async def query_search(
        self, query, index_name, top_n, min_threshold, openai_client
    ):
        try:
            response_url = []
            scores = []
            url_ids = []
            species_response = await self.Species_and_variety_query(
                query, openai_client=openai_client
            )
            species_response = eval(species_response)
            species = species_response.get("species", "None")
            variety = species_response.get("Variety_of_species", "None")
            if species == "None":
                species = None
            elif variety == "None":
                variety = None
            result = await self.search_ai(
                query,
                index_name,
                top_n=top_n,
                min_threshold=min_threshold,
                species=species,
                variety=variety,
                openai_client=openai_client,
            )
            for results in result:
                if results:
                    response_url.append(results["url"])
                    scores.append(results["@search.score"])
                    url_ids.append(results["url_id"])
            if not response_url:
                print("Searching again")
                result = await self.search_ai(
                    query, index_name, top_n, min_threshold=min_threshold
                )
                for results in result:
                    response_url.append(results["url"])
                    scores.append(results["@search.score"])
                    url_ids.append(results["url_id"])
            return response_url, scores, url_ids
        except Exception as e:
            raise Exception(
                f"Exception occured while fetching top {top_n} results: {e}"
            )

    async def search(self):
        res = await self.query_search(
            query=self.query,
            index_name=self.index_name,
            top_n=self.top_n,
            min_threshold=self.min_threshold,
            openai_client=openai_client,
        )
        return res[-1]


async def video_search(
    query: Annotated[str, "query of which video id needs to fetch"],
    top_n: Annotated[int, "n video_id retreivel"] = 3,
):
    """
    This tool returns the video id of ingested video corresponds to the query
    """
    video_search = VideoSearch(query=query, top_n=top_n)
    video_id = await video_search.search()
    return video_id


if __name__ == "__main__":
    query = "What is IPL 321?"
    res = asyncio.run(video_search(query=query))
    print(res)
