# DRL models from RLlib
import ray
from ray.rllib.algorithms.a3c import A3C
from ray.rllib.algorithms.ddpg import DDPG
from ray.rllib.algorithms.td3 import TD3
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC

MODELS = {"a2c": A3C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

from ray.tune.registry import register_env

# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}


class Rllib_model:
    def __init__(self, trainer):
        self.trainer = trainer

    def __call__(self, state):
        return self.trainer.compute_single_action(state)


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

        model = MODELS[model_name]
        # get algorithm default configration based on algorithm in RLlib
        if model_config is None:
            if model_name == "a2c":
                model_config = A3C.get_default_config().copy()
            elif model_name == "td3":
                model_config = TD3.get_default_config().copy()
            else:
                model_config = MODELS[model_name].get_default_config().copy()

        register_env("finrl_env", self.env)
        model_config["env"] = "finrl_env"
        model_config["env_config"] = env_config
        model_config["log_level"] = "WARN"
        model_config["framework"] = framework

        return model, model_config

    def train_model(
        self,
        model,
        model_name,
        model_config,
        total_episodes=100,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        trainer = MODELS[model_name](config=model_config)

        cwd = "./test_" + str(model_name)
        for _ in range(total_episodes):
            trainer.train()
            # save the trained model
            trainer.save(cwd)

        ray.shutdown()

        return trainer

    @staticmethod
    def DRL_prediction(
        model_name,
        env,
        env_config,
        agent_path="./test_ppo/checkpoint_000100/checkpoint-100",
        init_ray=True,
        model_config=None,
        framework="torch",
    ):
        if init_ray:
            ray.init(
                ignore_reinit_error=True
            )  # Other Ray APIs will not work until `ray.init()` is called.

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_config is None:
            model_config = MODELS[model_name].get_default_config().copy()

        register_env("finrl_env", env)
        model_config["env"] = "finrl_env"
        model_config["env_config"] = env_config
        model_config["log_level"] = "WARN"
        model_config["framework"] = framework

        trainer = MODELS[model_name](config=model_config)

        try:
            trainer.restore(agent_path)
            print("Restoring from checkpoint path", agent_path)
        except BaseException:
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
        model_config = MODELS[model_name].get_default_config().copy()
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

        trainer = MODELS[model_name](config=model_config)

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

        model_config = MODELS[model_name].get_default_config().copy()
        model_config["env"] = env
        model_config["log_level"] = "WARN"

        env_instance = env

        trainer = MODELS[model_name](config=model_config)

        try:
            trainer.restore(agent_path)
            print("Restoring from checkpoint path", agent_path)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        state = env_instance.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [env_instance.initial_total_asset]
        done = False
        while not done:
            action = trainer.compute_single_action(state)
            state, reward, done, _ = env_instance.step(action)

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
