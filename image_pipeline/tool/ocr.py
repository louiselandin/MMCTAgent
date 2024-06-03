from models.VIT.ocr import TROCRBase, TROCRLarge, TROCRSmall
from llama_index.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.tools.utils import create_schema_from_function



class OCRTool(AsyncBaseTool):
    def __init__(self):
        self.tools = { "1": TROCRSmall(),
                       "2": TROCRBase(),
                       "3": TROCRLarge()}
        
        self._metadata = ToolMetadata(self.get_desc(), "ocr", create_schema_from_function("ocr", self.call_as_fn , additional_fields=None ))
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
                Optical Character Recognition Tool: ocr\n
               """
    def get_desc(self):
        return """
                You can use this tool to analyze the given image, The tool should be used when
                you require to extract text from the image. The algorithm returns
                the extracted text which might not be accurate given the limited performance of the OCR model.

                This returns response in a list of strings which is simply in the order of the 
                text present in the image from left to right and top to bottom.
                input: 
                    {}
                Input is always empty as it doesnt require anything as input and analyzes on the image that you are given. Always ignore the arguement priority and do not generate that in the input.

                response:
                    The output is a list of string containing the text that is extracted in the order it is present in the image.
               """
    def get_fn_schema(self):
        return """
                priority: Optional[str] = "3" \n \t possible values: ["1","2","3"]
               """
    def __str__(self):
        return f"""
                {self.get_name()}
                {self.get_desc()}
               """
        
        