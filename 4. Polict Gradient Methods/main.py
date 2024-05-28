import os
import pickle
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import optuna

gc.enable()

TRAIN = True
FINETUNE = False
SAVE_MODEL = True  # Save the model after training
SAVE_VIDEO = True  # Save the video of the training process
RENDER_MODE = 'rgb_array' # Set to 'human' to render the environment

# Training hyperparameters
TRAINING_EPISODES = 1  # valid only if TRAIN is True
FINETUNE_TRIALS = 100  # valid only if FINETUNE is True

# Set the following hyperparameters if FINETUNE is False
GAMMA = 0.99
LEARNING_RATE = 1e-3

# Define the policy network architecture
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device=torch.device('cpu')):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class REINFORCE:
    def __init__(self, env, policy_network, optimizer, model_path='model/model.pth', gamma=0.99):
        self.env = env
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.model_path = model_path
        self.gamma = gamma

        # Load the model if it exists
        if os.path.exists(os.path.dirname(self.model_path)):
            if os.path.isfile(self.model_path):
                self.policy_network.load_state_dict(torch.load(self.model_path))
                print("Loaded model from disk")
        else:
            os.makedirs(os.path.dirname(self.model_path))

    def train(self, num_episodes, save_model=SAVE_MODEL, save_video=SAVE_VIDEO):
        total_rewards = []

        # Create a VideoWriter to save the rendering
        if save_video:
            self.video = VideoRecorder(self.env, f'{os.path.dirname(__file__)}/training.mp4', enabled=True)

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            log_probs = []
            rewards = []

            while not done:
                if RENDER_MODE == 'human':
                    self.env.render()

                if save_video:
                    self.video.capture_frame()

                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs = self.policy_network(state)
                action = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs.squeeze(0)[action])
                log_probs.append(log_prob)

                next_state, reward, done, _, _ = self.env.step(action)
                rewards.append(reward)
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            self.update_policy(log_probs, rewards)

            print(f"Episode {episode}, Total Reward: {total_reward}")

            if episode % 5 == 0 and episode > 0:
                print(f"Episode {episode}, Average Reward: {sum(total_rewards) / len(total_rewards)}")
                if save_model:
                    torch.save(self.policy_network.state_dict(), self.model_path)
                    print("Saved model to disk")
            
            # Delete variables to free up memory
            del log_probs, rewards, state, action_probs, action
            gc.collect()

        # Save the model after training
        if save_model:
            torch.save(self.policy_network.state_dict(), self.model_path)
            print("Saved model to disk")

        if save_video:
            self.video.close()
        
        self.env.close()
        return sum(total_rewards) / len(total_rewards)  # Return average reward

    def update_policy(self, log_probs, rewards):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = sum(self.gamma ** i * rewards[t + i] for i in range(len(rewards) - t))
            discounted_rewards.append(Gt)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * Gt)
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

class Optimizer:
    def __init__(self, env, policy_network, model_path, params_path='params.pkl'):
        self.env = env
        self.policy_network = policy_network
        self.model_path = model_path
        self.params_path = params_path

    def objective(self, trial, n_episodes=100):
        # Suggest values for the hyperparameters
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        gamma = trial.suggest_uniform('gamma', 0.9, 0.999)

        # Create a new optimizer with the current learning rate
        optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Create a new trainer with the current hyperparameters
        trainer = REINFORCE(self.env, self.policy_network, optimizer, self.model_path, gamma=gamma)

        # Train the model and get the average reward
        reward = trainer.train(n_episodes, save_model=False, save_video=False)

        return reward

    def optimize(self, n_trials=100, save_params=True):
        # Load the parameters if they exist
        if not TRAIN and os.path.isfile(self.params_path):
            with open(self.params_path, 'rb') as f:
                best_params = pickle.load(f)
            print("Loaded parameters from disk")
        elif not FINETUNE:
            best_params = {
                'lr': LEARNING_RATE, 
                'gamma': GAMMA
                }
            print(f"Using default parameters: {best_params}")
        else:
            print("Optimizing hyperparameters")
            study = optuna.create_study(direction='maximize')
            study.optimize(self.objective, n_trials=n_trials)
            best_params = study.best_params

            # Save the parameters if requested
            if save_params:
                with open(self.params_path, 'wb') as f:
                    pickle.dump(best_params, f)
                print("Saved parameters to disk")

        return best_params

# Initialize environment, networks, optimizer, and replay buffer
env = gym.make('MountainCar-v0', render_mode=RENDER_MODE)  # Use the MountainCar-v0 environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
policy_network = PolicyNetwork(state_dim, action_dim, device=device)

optimizer = Optimizer(env, policy_network, f'{os.path.dirname(__file__)}/model/model.pth', f'{os.path.dirname(__file__)}/model/params.pkl')
best_params = optimizer.optimize(n_trials=FINETUNE_TRIALS, save_params=True)

optimizer = optim.Adam(policy_network.parameters(), lr=best_params['lr'])
trainer = REINFORCE(env, policy_network, optimizer, f'{os.path.dirname(__file__)}/model/model.pth', gamma=best_params['gamma'])
trainer.train(TRAINING_EPISODES, save_model=SAVE_MODEL, save_video=SAVE_VIDEO)

