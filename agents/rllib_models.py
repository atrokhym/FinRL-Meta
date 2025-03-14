# DRL models from RLlib
import ray
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

# Import the algorithm classes
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC

# Map algorithm names to their classes and configs
MODELS = {
    "sac": {"class": SAC, "config": SACConfig},
    "ppo": {"class": PPO, "config": PPOConfig}
}

from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}


class Rllib_model:
    def __init__(self, trainer):
        self.trainer = trainer
        # Get the RLModule instance
        self.module = trainer.get_module("default_policy")

    def __call__(self, state):
        # Convert state to a PyTorch tensor batch of size 1
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            # If it's a list or other type, convert to numpy first, then to tensor
            state_tensor = torch.tensor([state], dtype=torch.float32)
            
        # Use the new API to compute actions
        output = self.module.forward_inference({"obs": state_tensor})
        
        # Print the output keys to debug
        print("Output keys:", list(output.keys()))
        
        # In Ray 2.43.0, the action key might be different
        if "actions" in output:
            action = output["actions"][0]
        elif "action" in output:
            action = output["action"][0]
        elif "action_dist_inputs" in output:
            # If we have action distribution inputs, we can sample from it
            action_dist_inputs = output["action_dist_inputs"]
            # For discrete actions, take the argmax
            action = torch.argmax(action_dist_inputs, dim=-1)[0]
        else:
            # If we can't find the action, print all keys and raise an error
            print("Available keys in output:", list(output.keys()))
            raise KeyError("Could not find action in output")
        
        # Convert action to numpy if it's a tensor
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        return action


class DRLAgent:
    """Implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, init_ray=True):
        self.env = env
        if init_ray:
            ray.init(
                ignore_reinit_error=True
            )  # Other Ray APIs will not work until `ray.init()` is called.

    def get_model(
        self,
        model_name,
        env_config,
        model_config=None,
        framework="torch",
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        model_info = MODELS[model_name]
        
        # Create a new config instance if none provided
        if model_config is None:
            model_config = model_info["config"]()
            
        # Register the environment
        register_env("finrl_env", self.env)
        
        # Configure the model
        model_config.environment(env="finrl_env")
        model_config.env_config = env_config
        model_config.log_level = "WARN"
        model_config.framework(framework)

        return model_info["class"], model_config

    def train_model(
        self,
        model_class,
        model_name,
        model_config,
        total_episodes=100,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        # Create the algorithm instance
        trainer = model_class(config=model_config)

        import os
        cwd = os.path.abspath("./test_" + str(model_name))
        for _ in range(total_episodes):
            trainer.train()
            # save the trained model
            try:
                trainer.save(cwd)
            except Exception as e:
                print(f"Warning: Could not save model: {e}")

        ray.shutdown()

        return trainer

    @staticmethod
    def DRL_prediction(
        model_name,
        env,
        env_config={},
        agent_path="./test_ppo/checkpoint_000100/checkpoint-100",
        init_ray=True,
        model_config=None,
        framework="torch",
    ):
        import os
        # Convert to absolute path if it's a relative path
        if agent_path.startswith("./") or agent_path.startswith("../"):
            agent_path = os.path.abspath(agent_path)
        if init_ray:
            ray.init(
                ignore_reinit_error=True
            )  # Other Ray APIs will not work until `ray.init()` is called.

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        model_info = MODELS[model_name]
        
        # Create a new config instance if none provided
        if model_config is None:
            model_config = model_info["config"]()
            
        # Register the environment
        register_env("finrl_env", env)
        
        # Configure the model
        model_config.environment(env="finrl_env")
        model_config.env_config = env_config
        model_config.log_level = "WARN"
        model_config.framework(framework)

        # Create the algorithm instance
        trainer = model_info["class"](config=model_config)

        try:
            trainer.restore(agent_path)
            print("Restoring from checkpoint path", agent_path)
        except BaseException as e:
            print(f"Error restoring from checkpoint: {e}")
            raise ValueError("Fail to load agent!")

        agent = Rllib_model(trainer)

        return agent


class DRLAgent_old:
    """Implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class
        price_array: numpy array
            OHLC data
        tech_array: numpy array
            techical data
        turbulence_array: numpy array
            turbulence/risk data
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array, turbulence_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array

    def get_model(
        self,
        model_name,
        # policy="MlpPolicy",
        # policy_kwargs=None,
        # model_kwargs=None,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        # if model_kwargs is None:
        #    model_kwargs = MODEL_KWARGS[model_name]

        model = MODELS[model_name]
        # get algorithm default configration based on algorithm in RLlib
        if model_name == "a2c":
            model_config = model.A2C_DEFAULT_CONFIG.copy()
        elif model_name == "td3":
            model_config = model.TD3_DEFAULT_CONFIG.copy()
        else:
            model_config = model.DEFAULT_CONFIG.copy()
        # pass env, log_level, price_array, tech_array, and turbulence_array to config
        model_config["env"] = self.env
        model_config["log_level"] = "WARN"
        model_config["env_config"] = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }

        return model, model_config

    def train_model(
        self,
        model,
        model_name,
        model_config,
        total_episodes=100,
        init_ray=True,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        if init_ray:
            ray.init(
                ignore_reinit_error=True
            )  # Other Ray APIs will not work until `ray.init()` is called.

        if model_name == "ppo":
            trainer = model.PPOTrainer(env=self.env, config=model_config)
        elif model_name == "a2c":
            trainer = model.A2CTrainer(env=self.env, config=model_config)
        elif model_name == "ddpg":
            trainer = model.DDPGTrainer(env=self.env, config=model_config)
        elif model_name == "td3":
            trainer = model.TD3Trainer(env=self.env, config=model_config)
        elif model_name == "sac":
            trainer = model.SACTrainer(env=self.env, config=model_config)

        for _ in range(total_episodes):
            trainer.train()

        ray.shutdown()

        # save the trained model
        cwd = "./test_" + str(model_name)
        trainer.save(cwd)

        return trainer

    @staticmethod
    def DRL_prediction(
        model_name,
        env,
        agent_path="./test_ppo/checkpoint_000100/checkpoint-100",
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        model_info = MODELS[model_name]
        model_config = model_info["config"]()
        model_config.environment(env=env)
        model_config.log_level = "WARN"

        env_instance = env

        # Create the algorithm instance
        trainer = model_info["class"](config=model_config)

        try:
            trainer.restore(agent_path)
            print("Restoring from checkpoint path", agent_path)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # Get the RLModule instance
        module = trainer.get_module("default_policy")
        
        # test on the testing env
        state, _ = env_instance.reset()  # Gymnasium returns (obs, info)
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [env_instance.initial_total_asset]
        done = False
        while not done:
            # Convert state to a PyTorch tensor batch of size 1
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            else:
                # If it's a list or other type, convert to numpy first, then to tensor
                state_tensor = torch.tensor([state], dtype=torch.float32)
                
            # Use the new API to compute actions
            output = module.forward_inference({"obs": state_tensor})
            
            # Print the output keys to debug
            print("Output keys:", list(output.keys()))
            
            # In Ray 2.43.0, the action key might be different
            if "actions" in output:
                action = output["actions"][0]
            elif "action" in output:
                action = output["action"][0]
            elif "action_dist_inputs" in output:
                # If we have action distribution inputs, we can sample from it
                action_dist_inputs = output["action_dist_inputs"]
                # For discrete actions, take the argmax
                action = torch.argmax(action_dist_inputs, dim=-1)[0]
            else:
                # If we can't find the action, print all keys and raise an error
                print("Available keys in output:", list(output.keys()))
                raise KeyError("Could not find action in output")
            
            # Convert action to numpy if it's a tensor
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            state, reward, done, truncated, _ = env_instance.step(action)  # Gymnasium returns (obs, reward, terminated, truncated, info)
            done = done or truncated  # Either terminated or truncated means the episode is done

            total_asset = (
                env_instance.amount
                + (env_instance.price_ary[env_instance.day] * env_instance.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / env_instance.initial_total_asset
            episode_returns.append(episode_return)
        ray.shutdown()
        print("episode return: " + str(episode_return))
        print("Test Finished!")
        return episode_total_assets
