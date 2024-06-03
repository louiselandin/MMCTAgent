from env.react_env.react_env import ReactEnv
from tool import BLIPTool

class BlipReactEnv(ReactEnv):
    def __init__(self, *args, **kwargs):
        tools = [BLIPTool()]
        super().__init__(tools, *args, **kwargs)
        
    def img_query(self, query, selected_img="both"):
        return self.tools[0].call(query, selected_img)
    
    def reset(self, new_images):
        self.tools[0].set_images(new_images)
        self.react.reset()

if __name__ == "__main__":
    
    env = BlipReactEnv()
    env.reset(images)