from multiprocessing import Pool
import numpy as np

class HEURReward:
    def __init__(self, max_thread=None):
        self.MAX_THREAD = max_thread or 16

    @staticmethod
    def compute_pair(self, pair):
        goal_text = pair["goal"].lower()
        action_text = pair["action"].lower()
        if goal_text.find("query")>=0:
            if action_text.find("action:")>=0 and action_text.find("action_input:")>=0:
                return 1
            else:
                return -1
        else:
            return 0

    def compute_batch(self, batch_of_pairs):
        batch_of_pairs = batch_of_pairs.tolist()
        with Pool(self.MAX_THREAD) as p:
            rewards = p.map(self.compute_pair, batch_of_pairs)
        return np.array(rewards)