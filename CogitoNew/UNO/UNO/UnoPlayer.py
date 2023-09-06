import wandb

# from DQNAgent import DQNAgent
import torch
import gymnasium as gym
import numpy as np
import rlcard
from rlcard.games.uno.game import UnoGame as Game

#This is the class where the training happens
def train_agent(agent):
    
    wordToActionInt = {"r-0": 0, "r-1": 1, "r-2": 2, "r-3": 3, "r-4": 4, "r-5": 5, "r-6": 6, "r-7": 7, "r-8": 8, "r-9": 9, "r-skip": 10, "r-reverse": 11, "r-draw_2": 12, "r-wild": 13, "r-wild_draw_4": 14, "g-0": 15, "g-1": 16, "g-2": 17, "g-3": 18, "g-4": 19, "g-5": 20, "g-6": 21, "g-7": 22, "g-8": 23, "g-9": 24, "g-skip": 25, "g-reverse": 26, "g-draw_2": 27, "g-wild": 28, "g-wild_draw_4": 29, "b-0": 30, "b-1": 31, "b-2": 32, "b-3": 33, "b-4": 34, "b-5": 35, "b-6": 36, "b-7": 37, "b-8": 38, "b-9": 39, "b-skip": 40, "b-reverse": 41, "b-draw_2": 42, "b-wild": 43, "b-wild_draw_4": 44, "y-0": 45, "y-1": 46, "y-2": 47, "y-3": 48, "y-4": 49, "y-5": 50, "y-6": 51, "y-7": 52, "y-8": 53, "y-9": 54, "y-skip": 55, "y-reverse": 56, "y-draw_2": 57, "y-wild": 58, "y-wild_draw_4": 59, "draw": 60}
    intToActionWord = {v: k for k, v in wordToAction.items()}
    
    print(intToAction)

    train_env = rlcard.make(agent.cfg.env)
    eval_env = rlcard.make(agent.cfg.env)

    # Logging
    wandb.init(project=agent.cfg.wandb_name, config=agent.cfg.get_members())

    for episode in range(1, agent.cfg.episodes + 1):
        
        obsdict, playerID = train_env.reset()
        episode_return = 0
        episode_lenght = 0
        losses = []

        while True:
            
            action = agent.act(obs)
            obsdict, playerID = train_env.step(action)
            
            truncated = False


            episode_return += reward
            episode_lenght += 1

            # save transition
            agent.store_transition(ob=obs, ac=action, rew=reward, next_ob=next_obs, done=truncated or terminated)
            # update DQN
            loss = agent.update_q_values()

            if loss is not None:
                losses.append(loss.item())

            obs = next_obs

            if terminated or truncated:
                break

        if not losses:
            wandb.log({"training return": episode_return, "train episode length": episode_lenght})
        else:
            losses = np.average(losses)
            wandb.log({"training return": episode_return,
                       "train episode length": episode_lenght,
                       "loss": losses})

        print("Episode", episode, "episode return:", episode_return, end="\t")

        # Update the target network
        if episode % agent.cfg.update_target_network_freq == 0:
            agent.update_target_network()

        if episode % agent.cfg.eval_freq == 0:

            obs, info = eval_env.reset()
            episode_return = 0
            episode_lenght = 0

            while True:
                
                action = agent._greedy_action(obs)
                next_obs, reward, terminated, info = eval_env.step(action)
                truncated = False
                
                obs = next_obs

                episode_return += reward
                episode_lenght += 1

                if terminated or truncated:
                    break
            wandb.log({"eval return": episode_return, "eval episode length": episode_lenght})
            print("Eval return:", episode_return, end="")
        print()
    wandb.finish()


if __name__ == '__main__':
    # agent = DQNAgent(DQNAgent.Config())
    # train_agent(agent)

    # agent.save("testdqn.pyt")

    wordToAction = {"r-0": 0, "r-1": 1, "r-2": 2, "r-3": 3, "r-4": 4, "r-5": 5, "r-6": 6, "r-7": 7, "r-8": 8, "r-9": 9, "r-skip": 10, "r-reverse": 11, "r-draw_2": 12, "r-wild": 13, "r-wild_draw_4": 14, "g-0": 15, "g-1": 16, "g-2": 17, "g-3": 18, "g-4": 19, "g-5": 20, "g-6": 21, "g-7": 22, "g-8": 23, "g-9": 24, "g-skip": 25, "g-reverse": 26, "g-draw_2": 27, "g-wild": 28, "g-wild_draw_4": 29, "b-0": 30, "b-1": 31, "b-2": 32, "b-3": 33, "b-4": 34, "b-5": 35, "b-6": 36, "b-7": 37, "b-8": 38, "b-9": 39, "b-skip": 40, "b-reverse": 41, "b-draw_2": 42, "b-wild": 43, "b-wild_draw_4": 44, "y-0": 45, "y-1": 46, "y-2": 47, "y-3": 48, "y-4": 49, "y-5": 50, "y-6": 51, "y-7": 52, "y-8": 53, "y-9": 54, "y-skip": 55, "y-reverse": 56, "y-draw_2": 57, "y-wild": 58, "y-wild_draw_4": 59, "draw": 60}

    intToAction = {v: k for k, v in wordToAction.items()}
    print()
    print(intToAction)