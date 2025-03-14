"""
Example script demonstrating how to use the updated RLlib models with Ray 2.43.0
"""
import ray
import gymnasium as gym
import numpy as np
from agents.rllib_models import DRLAgent, MODELS

# Create a simple environment
class SimpleEnv(gym.Env):
    def __init__(self, config=None):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.state = np.zeros(4, dtype=np.float32)
        self.episode_length = 0
        self.max_episode_length = 10
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-0.1, 0.1, size=4).astype(np.float32)
        self.episode_length = 0
        return self.state, {}  # Gymnasium requires returning info dict
        
    def step(self, action):
        self.state = np.clip(self.state + np.random.uniform(-0.1, 0.1, size=4), -1, 1).astype(np.float32)
        reward = float(action)  # Simple reward: action 1 gives reward 1, action 0 gives reward 0
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return self.state, reward, done, False, {}  # Gymnasium requires info and truncated

def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Create environment creator function
    def env_creator(config=None):
        return SimpleEnv(config)

    # Create agent
    agent = DRLAgent(env=env_creator)

    # Get model
    model_name = "ppo"  # Use PPO since it's available
    print(f"Using model: {model_name}")

    model_class, model_config = agent.get_model(
        model_name=model_name,
        env_config={},
        framework="torch"
    )

    print("Model configuration created successfully")

    # Train for just 2 episodes to test
    print("Training model for 2 episodes...")
    trainer = agent.train_model(
        model_class=model_class,
        model_name=model_name,
        model_config=model_config,
        total_episodes=2
    )

    print("Training completed successfully")

    # Test prediction
    print("Testing prediction...")
    import os
    agent_path = os.path.abspath(f"./test_{model_name}")
    test_agent = DRLAgent.DRL_prediction(
        model_name=model_name,
        env=env_creator,
        agent_path=agent_path
    )

    # Test the agent
    env = SimpleEnv()
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = test_agent(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    print(f"Test episode completed with total reward: {total_reward}")

    # Clean up
    ray.shutdown()
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
