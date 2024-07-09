from models.VIT.gpt4v import GPT4V
from llama_index.tools.types import AsyncBaseTool, ToolMetadata, ToolOutput
from llama_index.tools.utils import create_schema_from_function

class CriticTool(AsyncBaseTool):
    def __init__(self):
        #super(BLIPTool).__init__()
        self.tools = { "3": GPT4V() }
        
        self._metadata = ToolMetadata(self.get_desc(), "critic", create_schema_from_function("critic", self.call_as_fn , additional_fields=None ))
        self.async_call =  self.sync_to_async(self.call_as_fn)
        self._image = {}
        self._conversation = ""
        self._query = ""
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
    def call_as_fn(self, priority:str = "3", query:str = None, conversation:str = None, **kwargs):
        if query:
            self._query = query
        if conversation:
            self._conversation = conversation
            
        prompt = f"""
                You are a critic for a vision language pipeline, The pipeline consists
                of a LLM comprehending a query along with image input. The LLM is able
                to use different tools to understand the image input. It is very critical
                to analyze 2 things, 1) Efficacy in tool usage and its performance
                for the subtask, 2) LLMs utilization for these observation and reasoning
                based on it.

                For doing so you are given a the previous conversation along with main 
                query
                ----------------------------------------------------------------------
                query: {self._query}
                conversation: {self._conversation}
                ----------------------------------------------------------------------
                I want a concise report which contains 4 checkboxes specified below

                - [ ] The First checkbox denotes if the conversation has answered the
                    original query completely or even partially
                - [ ] Understand how the tools are used and decomposed into subtasks and 
                    if They utilize all relevant information available for the query.
                    You have to take a good look into the image you are given and assert
                    if the LLM was presented with all necessary information.
                - [ ] This is to understand any discrepancies in the reasoning chain by
                    the LLM in the conversation, You have to verify that all the steps 
                    and raise concerns if the facts are incorrect.
                - [ ] Apart from above points if you find any other scope of improvement
                    please suggest it to the LLM. And collecting all the three points
                    finally draft a Feedback for the LLM to improve the reasoning for the
                    task.

                You have to go through them step by step and finally format them as shown

                - [X] Answered
                - [ ] All information used
                - [ ] Verification of conversation
                - [ ] Feedback

                The checkboxes should be filled based on the condition given above. Feedback
                checkbox is filled when you believe that the conversation is correct in all the
                above evaluation methods and when you cannot find any mistake in the conversation

                In the above conversation you may see a critic verification make sure you assert 
                those feedbacks and if they are rectified by the LLM.  
                """
        return self.tools[priority](prompt, self._image[self.idx])
        
    def set_image(self, image, idx=0):
        self._image[idx] = image

    def set_conversation(self, conversation):
        self._conversation = conversation

    def set_query(self, query):
        self._query = query

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
                Image Recognition Tool: Critic\n
               """
    def get_desc(self):
        return """
                You are supposed to call this tool before giving the final response to answer the question. 
                You have to use this tool irrespecitve of if you can answer the question without a tool, do not
                use it in between the reasoning chain only at the end. 
                This tool will evaluate the answer and provide feedback on the answer.
                input:
                    {}
                    
                    The critic has access to all the information about the React agent and its actions.
                    It also has access to the question and the image for the query.

                Your task is to call it before the end of the reasoning chain that is before final response, and only give the final response if all criteria 
                in the critic is satisfied else take the feedback and continue the chain with tools and followed by critic prior to the final response,
                You have to use the feedback to improve your action and solve the query efficiently. It is very critical to meet all criteria before giving the final response.
                
                response:
                    The output is simple text giving feedback and checkboxes based on evaluation criteria.
               """
    def get_fn_schema(self):
        return """
               None
               """
    def __str__(self):
        return f"""
                {self.get_name()}
                {self.get_desc()}
               """
        