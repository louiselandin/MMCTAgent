from pipeline import React
from rewards import HEURReward
from copy import deepcopy
# ToDo: Implement using Gym.Env
from contextlib import contextmanager

@contextmanager
def manage_id(tools, idx=0):
    for tool in tools:
        tool.idx = idx
    try:
        yield tool
    finally:
        for tool in tools:
            tool.idx = 0
        

class ReactEnv:
    def __init__(self, tools, num_llms, *args, **kwargs):
        self.tools = tools
        # for i in range(num_llms):
        #     self.tools.append(deepcopy(tools))
        self.react = React(self.tools, *args, **kwargs)
        
    def step(self, action_str, tries=10, idx=0):
        original_tries = tries
        resp = ""
        #while tries>0:
        with manage_id(self.tools, idx=idx):
            resp = self.react.chat(action_str, idx=idx)
        # resp = self.react.chat(action_str, idx=idx)
        #    continue
            # try:
            #     resp = self.react.chat(action_str)
            #     continue
            # except Exception as e:
            #     print(e)
            #     self.react.remove_last_step()
        #    tries -= 1
        done = tries == original_tries
        # dummy reward
        # chat_messages = self.react.fetch_last_step() or []
        # user_messages = []
        # assistant_messages = []
        # for message in chat_messages:
        #     if message.role == "user":
        #         user_messages.append(message.content)
        #     else:
        #         assistant_messages.append(message.content)
        # resp = [user_messages, assistant_messages]
        # reward = 1
        return resp # , reward, done, {}

    def reset(self, new_args_for_tools=None, idx=0):
        if new_args_for_tools:
            for tool in self.tools:
                tool.set_new_args(new_args_for_tools, idx=idx)
        self.react.reset(idx=idx)

