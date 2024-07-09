from env.react_env.react_env import ReactEnv
from tool import OCRTool, RECOGTool, ObjectDetectTool, VITTool, CriticTool
from utils.content_moderation import safe_pipeline_execute, safe_step_execute

class ToolReactEnv(ReactEnv):
    def __init__(self, *args, **kwargs):
        tools = [VITTool(), CriticTool(), OCRTool(), RECOGTool(), ObjectDetectTool()]
        super().__init__(tools, *args, **kwargs)
        
    @safe_pipeline_execute(class_func=True)
    def reset(self, new_image, new_question, new_identifier, idx=0):
        for i in range(len(self.tools)):
            self.tools[i].set_image(new_image, idx=idx)
            if hasattr(self.tools[i], "set_query"):
                self.tools[i].set_query(new_question)
            if hasattr(self.tools[i], "set_identifier"):
                self.tools[i].set_identifier(new_identifier)
        self.react.reset(idx=idx)

if __name__ == "__main__":
    
    env = ReactEnv()