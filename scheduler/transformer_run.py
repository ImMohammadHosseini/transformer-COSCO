

"""
"""

from .Scheduler import *
from .TRL.train import load_model
from .TRL.src.models import TransformerScheduler, EncoderScheduler

import pickle
import torch
from os import path, mkdir
from copy import deepcopy
from random import sample, randint
import numpy as np

class TRLScheduler(Scheduler):
    def __init__(self, data_type, training=True):
        self.save_path = 'scheduler/TRL/checkpoints'
        
        self.data_type = data_type
        self.hosts = int(data_type.split('_')[-1])
        self.encoder_max_length = int(3*self.hosts) + 3
        self.decoder_max_length = int(1*self.hosts) + 2
        self.prob_len = int(2*self.hosts**2)
        self.model = EncoderScheduler(self.encoder_max_length, 
                                      self.decoder_max_length, 3, 32,
                                      self.prob_len, self.hosts)
        
        self.model = load_model(self.save_path, self.model)
        #dtl = data_type.split('_')
        #print("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")

        #_, _, self.max_container_ips = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")

        
    def run_transformer(self):
        SOD = [[1]*3]
        EOD = [[2]*3]
        PAD = [0]*3       
        #TODO add more info
        contInfo = [(c.getApparentIPS()/50000, c.getHostID()/10 , (c.createAt+1)/self.env.interval) if c else (0,0,0) for c in self.env.containerlist]#0 if c.getHostID()==-1 else 1 #, c.getRAM()[0], c.getDisk()[0], c.createAt, c.startAt
        contInfo = np.array(contInfo)#self.padding(contInfo, int(1*self.hosts), PAD, 
                            #             pad_left=True))
        #50000
        hostInfo = np.array([(host.ipsCap/50000, host.getCPU()/100, host.latency) for host in self.env.hostlist])# , host.getRAMAvailable()[0], host.getDiskAvailable(), 0, 0
        mainInfo = np.append(np.append(np.append(SOD, contInfo, 0), EOD, 0), 
                             np.append(hostInfo, EOD, 0), 0)
        
        allocateInfo = [[1]*10]; prev_alloc = {}; PAD1 = [0]*10; step = 0
        #for c in self.env.containerlist:
        #    if c: hId = c.getHostID(); prev_alloc[c.id] = hId
        #    if c and hId != -1: 
        #        step += 1
        #        host = c.getHost()
        #        allocateInfo.append((c.getBaseIPS(), c.getRAM()[0], c.getDisk()[0], \
        #                              c.createAt, c.startAt, host.getIPSAvailable(), \
        #                                  host.getRAMAvailable()[0],  host.getDiskAvailable(), \
        #                                      0, 0))
        #allocateInfo = np.array(self.padding(allocateInfo, int(1.5*self.hosts)+1, 
        #                                     PAD1, pad_left=False))
        mainInfo = torch.tensor(mainInfo, dtype=torch.float32, 
                                requires_grad=True).unsqueeze(0)
        #print(mainInfo)
        #allocateInfo = torch.tensor(allocateInfo, dtype=torch.float32, 
        #                            requires_grad=True).unsqueeze(0)
        decisions, filter_decision, rewards, actions, log_probs, encoder_inputs, \
            steps, decoder_inputs = self.model.generateSteps(self.env, mainInfo, allocateInfo, step)
        
        return decisions, filter_decision, rewards, actions, log_probs, mainInfo, encoder_inputs, steps, decoder_inputs 
        
    def padding(self, sequence, final_length, padding_token, pad_left = True):
        pad_len = final_length - len(sequence)
        if pad_len == 0: return sequence
        pads = np.array([padding_token] * pad_len)
        if pad_left: return np.append(pads, np.array(sequence) , 0)  
        else: return np.append(np.array(sequence), pads, 0)   
    
    def selection(self):
        return []

    def placement(self, containerIDs):
        decision, _, _, _, _ = self.run_transformer()
        return decision