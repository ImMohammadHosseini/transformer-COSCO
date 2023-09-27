
"""

"""
import torch
from typing import Optional, List
import numpy as np

class PPOReplayBuffer ():
    def __init__ (self):
        self.reset()
    
    def reset (self):
        self.reset_mid_memory()
        self.reset_final_memory()
    
    def reset_final_memory (self) :
        self.final_observation = [] 
        self.final_internalObservation = []
        self.final_decision = [] 
        self.final_action = [] 
        self.final_prob = [] 
        self.final_val = []
        self.final_reward = []
        self.final_steps = []
        
    def reset_mid_memory (self) :
        self.mid_observation = [] 
        self.mid_internalObservation = []
        self.mid_decision = [] 
        self.mid_action = [] 
        self.mid_prob = [] 
        self.mid_val = []
        self.mid_steps = []
    
    def save_mid_memory (
        self, 
        observation: List[torch.Tensor],
        decision: List,
        rewards,
        actions: List,
        probs: List,
        values: List, 
        steps: Optional[List[torch.Tensor]] = None,
        internalObservations: Optional[List[torch.Tensor]] = None,
    ):
        self.mid_observation += observation
        self.mid_decision += decision
        self.mid_action += actions
        self.mid_prob += probs
        self.mid_val += values
        
        if not isinstance(steps, type(None)):
            self.mid_steps += steps
            
        if not isinstance(internalObservations, type(None)):
            self.mid_internalObservation += internalObservations
        
        self.save_final_memory(rewards)

    def save_final_memory (
        self,
        rewards:dict
    ):
        re = dict(reversed(list(rewards.items())))
        for i in re:
            condition = (np.array(self.mid_decision)[:,0] == i[0]) &  (np.array(self.mid_decision)[:,1] == i[1])
            #print(condition)
            #print(re[i])
            index = np.where(condition)[0][-1]
            self.final_observation.append(self.mid_observation.pop(index))
            self.final_decision.append(self.mid_decision.pop(index)) 
            self.final_action.append(self.mid_action.pop(index))
            self.final_prob.append(self.mid_prob.pop(index))
            self.final_val.append(self.mid_val.pop(index))
            self.final_reward.append(re[i])
            
            try:
                self.final_internalObservation.append(self.mid_internalObservation.pop(index))
                self.final_steps.append(self.mid_steps.pop(index))
            except: pass
        
    def get_memory (
        self, 
        n_state: int,
    ):
        try: return np.array(self.final_observation[:n_state]), \
                np.array(self.final_action[:n_state]), \
                np.array(self.final_prob[:n_state]), \
                np.array(self.final_val[:n_state]), \
                np.array(self.final_reward[:n_state]), \
                np.array(self.final_steps[:n_state]), \
                np.array(self.final_internalObservation[:n_state])
        except: return np.array(self.final_observation[:n_state]), \
                np.array(self.final_action[:n_state]), \
                np.array(self.final_prob[:n_state]), \
                np.array(self.final_val[:n_state]), \
                np.array(self.final_reward[:n_state])
                
    def erase(
        self, 
        n_state,
    ):
        try:
            self.final_observation = self.final_observation[n_state:]
            self.final_action = self.final_action[n_state:]
            self.final_prob = self.final_prob[n_state:]
            self.final_val = self.final_val[n_state:]
            self.final_reward = self.final_reward[n_state:]
            self.final_steps = self.final_steps[n_state:]
            self.final_internalObservation = self.final_internalObservation[n_state:]
        except: 
            self.final_observation = self.final_observation[n_state:]
            self.final_action = self.final_action[n_state:]
            self.final_prob = self.final_prob[n_state:]
            self.final_val = self.final_val[n_state:]
            self.final_reward = self.final_reward[n_state:]
        return len(self.final_action)
        