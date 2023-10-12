"""
"""
import torch
import numpy as np
import os
import pickle
from tqdm import tqdm

from .src.ppo_trainer import PPOTRainer

def ppo_train(workload, scheduler, datacenter, train_step):
    #env = scheduler.env
    trainer = PPOTRainer(scheduler.model)
    batch_size = 32
    episod_step = 100
    
    reward_history=[]; avgresponsetime_history=[]; energytotal_history=[]; num_container_history=[]
    for i in tqdm(range(train_step)):
        #TODO change reset repluy buffer
        #TODO CHANGE INPUT
        
        workload.reset()
        newcontainerinfos = workload.generateNewContainers(scheduler.env.interval) # New containers info
        hostlist = datacenter.generateHosts()
        scheduler.env.reset(hostlist)
        deployed = scheduler.env.addContainersInit(newcontainerinfos) # Deploy new containers and get container IDs
        decisions, filter_decision, rewards, actions, log_probs, mainInfo, encoder_inputs, \
            steps, decoder_inputs = scheduler.run_transformer()
        trainer.reset()
        trainer.save_mid_step (encoder_inputs, decisions, filter_decision, rewards,
                               actions, log_probs, steps, decoder_inputs)

        migrations, rewards = scheduler.env.allocateInit(filter_decision) # Schedule containers
        #print('rewards', rewards)
        trainer.save_final_step(rewards)
        #print(trainer.dd)

        workload.updateDeployedContainers(scheduler.env.getCreationIDs(migrations, deployed)) # Update workload allocated using creation IDs
        best_responsetime = 1e9
        ep_reward = sum(r for pr in rewards.values() for r in pr) 
        ep_avgresponsetime=[]; ep_totalenergy=0; num_container=0
        #print(rewards)
        n_steps = sum(len(pr) for pr in rewards.values())
        
        for ep in range(episod_step):
            
            #print(scheduler.env.getNumActiveContainers())
            newcontainerinfos = workload.generateNewContainers(scheduler.env.interval) 
            deployed, destroyed = scheduler.env.addContainers(newcontainerinfos)
            decisions, filter_decision, rewards, actions, log_probs, mainInfo, \
                encoder_inputs, steps, decoder_inputs = scheduler.run_transformer()
            n_steps += sum(len(pr) for pr in rewards.values())
            
            #print(rewards)
            ep_reward += sum(r for pr in rewards.values() for r in pr) 
            #filter_decisions = scheduler.filter_placement(decisions)
            trainer.save_mid_step (encoder_inputs, decisions, filter_decision, 
                                   rewards, actions, log_probs, steps, decoder_inputs)
        
            migrations, rewards = scheduler.env.simulationStep(filter_decision)
            #print(rewards)
            #print(trainer.dd)

            ep_reward += sum(r for pr in rewards.values() for r in pr) 
            ep_avgresponsetime += [c.totalExecTime + c.totalMigrationTime for c in destroyed] if len(destroyed) > 0 else []
            num_container +=len(destroyed)
            ep_totalenergy += np.sum([host.getPower()*scheduler.env.intervaltime for host in scheduler.env.hostlist])
            
            workload.updateDeployedContainers(scheduler.env.getCreationIDs(migrations, deployed)) 
            
            n_steps += sum(len(pr) for pr in rewards.values())
            trainer.save_final_step(rewards)
            if n_steps >= batch_size:
                n_steps = trainer.train_minibatch(batch_size)
        
        #num_container = workload.creation_id - scheduler.env.getNumActiveContainers()
        
        num_container_history.append(num_container)
        reward_history.append(ep_reward)
        ep_avgresponsetime = np.average(ep_avgresponsetime)
        avgresponsetime_history.append(ep_avgresponsetime)
        energytotal_history.append(ep_totalenergy)
        
        avg_responsetime= np.mean(avgresponsetime_history[-50:])
        if avg_responsetime < best_responsetime:
                best_responsetime  = avg_responsetime
                save_model(scheduler.save_path, trainer.actor_model, trainer.actor_optimizer)
                save_model(scheduler.save_path, trainer.critic_model, trainer.critic_optimizer)
        
        print('episod_reward %.3f' % ep_reward, 'avg_reward %.3f' % np.mean(reward_history[-50:]),
              'episod_container', num_container, 'avg_container %.3f' % np.mean(num_container_history[-50:]),
              'episod_avgresponsetime %.3f'%ep_avgresponsetime, 
              '50episod_avgresponsetime %.3f'% np.mean(avgresponsetime_history[-50:]), 
              'episod_totalenergy %.3f'%ep_totalenergy)
        
        if i % 100 == 0:
            results_dict = {'reward': reward_history, 'avgresponsetime': avgresponsetime_history,
                            'energytotal':energytotal_history, 'num_container':num_container_history}
            with open('scheduler/TRL/train_results/results.pickle', 'wb') as file:
                pickle.dump(results_dict, file)
    


def save_model(save_path, model, optimizer):
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    file_path = save_path + "/" + model.name + "_" + "TRL" + ".ckpt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, file_path)
    
def load_model(save_path, model):
    file_path = save_path + "/" + model.name + "_" + "TRL" + ".ckpt"
    if os.path.exists(file_path): 
        print("Loading pre-trained model: ")
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Creating new model: "+model.name)
    return model