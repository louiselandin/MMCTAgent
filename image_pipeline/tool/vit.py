from models.VIT.gpt4v import GPT4V
from llama_index.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.tools.utils import create_schema_from_function

class VITTool(AsyncBaseTool):
    def __init__(self):
        #super(BLIPTool).__init__()
        self.tools = { "3": GPT4V() }
        
        self._metadata = ToolMetadata(self.get_desc(), "vit", create_schema_from_function("vit", self.call_as_fn , additional_fields=None ))
        self.async_call =  self.sync_to_async(self.call_as_fn)
        self._image = {}
        self.idx = 0
    
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
    def call_as_fn(self, priority:str = "3"):
        return self.tools[priority](self._image[self.idx])
        
    def set_image(self, image, idx=0):
        self._image[idx] = image
        
    def sync_to_async(self, fn):
        """Sync to async."""

        async def _async_wrapped_fn(*args, **kwargs):
            return fn(*args, **kwargs)

        return _async_wrapped_fn
    async def acall(self, *args, **kwargs):
        tool_output = await self.async_call(*args, **kwargs)
        return ToolOutput(
            content = str(tool_output),
            tool_name = self.metadata.name,
            raw_input = {"args": args, "kwargs": kwargs},
            raw_output = tool_output,
        )
    
    def get_name(self):
        return """
                Image Recognition Tool: VIT\n
               """
    def get_desc(self):
        return """
                You can query information about the given image/images using simple natural language,
                This returns responses in simple language.
                input: 
                    {"query": "What is the number of objects in the image"}
                    or 
                    {"query": "What is the number of objects in the image", "selected_image": "1"}

                    The input can contain two values "query" and "selected_image". "selected_image" is optional but "query" is necessary for all queries.
                    "query" is to define the question that the Vision expert would answer about the image.
                    "selected_image" is used only when there are multiple images given in the problem setting. There are three valid options for "selected_image" i.e., "1", "2", "all". By default all is used, and for scenarios where there is only one image "selected_image" do not change the selection of image.

                response:
                    The output is simple text answering the query given.
               """
    def get_fn_schema(self):
        return """
               query: str
               selected_image: Optional[str] = "all" \n \t possible values: ["1","2",...(any number)...,"all"]
               """
    def __str__(self):
        return f"""
                {self.get_name()}
                {self.get_desc()}
               """
        