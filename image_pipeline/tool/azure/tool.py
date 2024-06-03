"""Tool for the Image Understanding search API."""

# from langchain.tools.base import BaseTool
from tool.azure.imun import ImunAPIWrapper
from llama_index.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.tools.utils import create_schema_from_function


class ImunRun(AsyncBaseTool):
    """Tool that adds the capability to query the Image Understanding API."""

    name = "Image Understanding"
    description = """A wrapper around Image Understanding. Useful for when you need to understand what is inside an image (objects, texts, people). Input should be an image url, or path to an image file (e.g. .jpg, .png)."""
    
    fn_schema_str_d = """
                    query: str should be path or url to an input image
                    """
    api_wrapper: ImunAPIWrapper

    def __init__(self):
        self._metadata = ToolMetadata(self.description, "imun", create_schema_from_function("imun", self.call_as_fn, additional_fields=None))
        self.async_call = self.sync_to_async(self.call_as_fn)

    @property
    def metadata(self):
        return self._metadata
    def call(self, *args, **kwargs):
        tool_output = self.call_as_fn(*args, **kwargs)

        return ToolOutput(
            content = str(tool_output),
            tool_name = self.metadata.name,
            raw_input = {"args": args, "kwargs": kwargs},
            raw_output = tool_output,
        )
    def call_as_fn(self, query):
        self._run(query)
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Image Understanding does not support async")


class ImunOCR:
    def __init__(self):
        self.imun = ImunAPIWrapper(
            imun_url=os.environ["IMUN_OCR_READ_URL"],
            params=os.environ["IMUN_OCR_PARAMS"],
            imun_subscription_key=os.environ["IMUN_SUBSCRIPTION_KEY"])
        
    def __call__(self, image):
        return self.imun.run(image)

class ImunObject:
    def __init__(self):
        self.imun = ImunAPIWrapper(
            imun_url=os.environ["IMUN_URL"],
            params=os.environ["IMUN_PARAMS"],
            imun_subscription_key=os.environ["IMUN_OCR_SUBSCRIPTION_KEY"])
        
    def __call__(self, image):
        return self.imun.run(image)

class ImunRecog:
    def __init__(self):
        self.imun = ImunAPIWrapper(
            imun_url=os.environ["IMUN_URL2"],
            params=os.environ["IMUN_PARAMS2"],
            imun_subscription_key=os.environ["IMUN_SUBSCRIPTION_KEY"])
        
    def __call__(self, image):
        return self.imun.run(image)

