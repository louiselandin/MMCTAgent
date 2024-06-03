from models.VIT.object_detect import YOLOs, DETARes, DETASwinL
from llama_index.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.tools.utils import create_schema_from_function

class ObjectDetectTool(AsyncBaseTool):
    def __init__(self):
        self.tools = { "1": YOLOs(),
                       "2": DETARes(),
                       "3": DETASwinL()}
        
        self._metadata = ToolMetadata(self.get_desc(), "object_detect", create_schema_from_function("object_detect", self.call_as_fn , additional_fields=None ))
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
                Object Detection Tool: object_detect\n
               """
    def get_desc(self):
        return """
                You can use this tool to analyze the given image, The tool should be used when
                individual objects are to be detected in the image. The algorithm returns
                positions of individual elements that it can detect.

                This returns response in a dictionary with the name of the object and the 
                position of the object in pixel coordinates in XYHW format.
                XYHW format represents 4 float values representing the X coordinate of the 
                object, Y coordinate of the object, the height of the object, Width of the object.
                input: 
                    {}
                Input is always empty as it doesnt require anything as input and analyzes on the image that you are given. Always ignore the arguement priority and do not generate that in the input.

                response:
                    The output is a dict containing object labels as key and a array in XYHW format corresponding the position of the object.
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
        
        