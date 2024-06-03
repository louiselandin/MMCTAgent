from models.VIT.instructblip import BlipT5XXL
from llama_index.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.tools.utils import create_schema_from_function

class BLIPTool(BlipT5XXL, AsyncBaseTool):
    def __init__(self):
        #super(BLIPTool).__init__()
        BlipT5XXL.__init__(self)
        self._metadata = ToolMetadata(self.get_desc(), "vit", create_schema_from_function("vit", self.call_as_fn , additional_fields=None ))
        self.async_call =  self.sync_to_async(self.call_as_fn)
    
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
    def call_as_fn(self, query:str, selected_image:str = "both"):
        return self(self._images, query, selected_image)
        
    def set_images(self, images):
        self._images = images
        
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
        
        