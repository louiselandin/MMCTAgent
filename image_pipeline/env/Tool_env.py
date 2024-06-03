from env.react_env.react_env import ReactEnv
from tool import OCRTool, RECOGTool, ObjectDetectTool, VITTool, CriticTool

class ToolReactEnv(ReactEnv):
    def __init__(self, *args, **kwargs):
        tools = [OCRTool(), RECOGTool(), ObjectDetectTool(), VITTool(), CriticTool()]
        super().__init__(tools, *args, **kwargs)
        
    def reset(self, new_image, idx=0):
        for i in range(len(self.tools)):
            self.tools[i].set_image(new_image, idx=idx)
        self.react.reset(idx=idx)

if __name__ == "__main__":
    
    env = BlipReactEnv()
    env.reset(images)