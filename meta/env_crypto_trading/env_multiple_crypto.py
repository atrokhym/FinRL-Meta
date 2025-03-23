import math
import gymnasium as gym
import numpy as np

class CryptoEnv(gym.Env):  # custom env
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        config,
        lookback=1,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        gamma=0.99,
    ):
        super().__init__()
        self.action_space = None  # Will be set in _setup_spaces
        self.observation_space = None  # Will be set in _setup_spaces
        self.lookback = lookback
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.max_stock = 1
        self.gamma = gamma
        self.price_array = np.array(config["price_array"])
        self.tech_array = np.array(config["tech_array"])
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback - 1

        # Define spaces
        self.state_dim = 1 + (self.price_array.shape[1] + self.tech_array.shape[1]) * lookback
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Properly initialize observation and action spaces"""
        self.action_space = gym.spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.price_array.shape[1],), 
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )

        # Initialize state
        self.time = self.lookback - 1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.episode_return = 0.0
        self.gamma_return = 0.0

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        self.time = self.lookback - 1
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        return self.get_state(), {}

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.time += 1
        price = self.price_array[self.time]

        # Process actions
        for i in range(self.action_space.shape[0]):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = actions[i] * norm_vector_i

        # Execute trades
        for index in np.where(actions < 0)[0]:
            if price[index] > 0:
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

        for index in np.where(actions > 0)[0]:
            if price[index] > 0:
                buy_num_shares = min(self.cash // price[index], actions[index])
                self.stocks[index] += buy_num_shares
                self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

        # Calculate reward and done
        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * price).sum()
        reward = (next_total_asset - self.total_asset) * 2**-16
        self.total_asset = next_total_asset
        
        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash
            
        return state, reward, done, False, {"episode_return": self.episode_return}

    def get_state(self):
        state = np.hstack((self.cash * 2**-18, self.stocks * 2**-3))
        for i in range(self.lookback):
            tech_i = self.tech_array[self.time - i]
            normalized_tech_i = tech_i * 2**-15
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)
        return state

    def close(self):
        pass

    def _generate_action_normalizer(self):
        action_norm_vector = []
        price_0 = self.price_array[0]
        for price in price_0:
            x = math.floor(math.log(max(price, 1e-8), 10))  # Prevent math domain error with minimum price value
            action_norm_vector.append(1 / ((10) ** x))
        self.action_norm_vector = np.asarray(action_norm_vector) * 10000



class MyCryptoEnv:  # custom env
    @property
    def amount(self):
        return self.cash

    @property
    def price_ary(self):
        return self.price_array

    @property
    def day(self):
        return self.time

    def __init__(self, config, lookback=1, initial_capital=1e6,
                 buy_cost_pct=1e-3, sell_cost_pct=1e-3, gamma=0.99):
        # Base initialization
        self.lookback = lookback
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.max_stock = 1
        self.gamma = gamma
        self.price_array = config['price_array']
        self.tech_array = config['tech_array']
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback - 1

        # Initialize equal-weight benchmark portfolio
        self.equal_weight_shares = np.array([initial_capital/(self.crypto_num * self.price_array[lookback-1, i])
                                           for i in range(self.crypto_num)])
        self.last_benchmark_value = initial_capital
        
        # Initialize risk metrics
        self.return_history = []
        self.volatility_window = 20
        self.volatility = 0.0
        
        # reset
        self.time = lookback-1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.episode_return = 0.0  
        self.gamma_return = 0.0
        
        # State space dimensions
        self.cash_dim = 1
        self.price_dim = self.price_array.shape[1]
        self.tech_dim = self.tech_array.shape[1]
        self.state_dim = 1 + 2 + 3 * self.price_dim + self.tech_dim
        
        # print(f"DEBUG: Dimension components:")
        # print(f"- cash_dim: {self.cash_dim}")
        # print(f"- price_dim: {self.price_dim}")
        # print(f"- tech_dim: {self.tech_dim}")
        # print(f"- lookback: {lookback}")
        # print(f"- Total state_dim: {self.state_dim}")

        '''env information'''
        self.env_name = 'MulticryptoEnv'
        self.action_dim = self.price_array.shape[1]
        self.if_discrete = False
        self.target_return = 10

    def reset(self):
        # Reset basic state
        self.time = self.lookback - 1
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        
        # Reset benchmark tracking
        self.equal_weight_shares = np.array([self.initial_cash/(self.crypto_num * self.price_array[self.time, i])
                                           for i in range(self.crypto_num)])
        self.last_benchmark_value = self.initial_cash
        
        # Reset risk metrics
        self.return_history = []
        self.volatility = 0.0
        
        state = self.get_state()
        info_dict = {
            'benchmark_value': self.last_benchmark_value,
            'volatility': self.volatility
        }
        return state, info_dict

    def step(self, actions):
        self.time += 1
        
        # Check if actions is a scalar and convert to array if needed
        if np.isscalar(actions) or (isinstance(actions, np.ndarray) and actions.ndim == 0):
            actions = np.full(self.action_dim, float(actions))
        else:
            # Create a copy of actions to avoid modifying the original
            actions = np.array(actions).copy()
        
        price = self.price_array[self.time]
        
        # Apply action normalization
        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = float(actions[i]) * norm_vector_i
            
        # Handle sells
        for index in np.where(actions < 0)[0]:
            if price[index] > 0:
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                
        # Handle buys
        for index in np.where(actions > 0)[0]:
            if price[index] > 0:
                buy_num_shares = min(self.cash // price[index], actions[index])
                self.stocks[index] += buy_num_shares
                self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

        # Update state
        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()

        # Calculate benchmark performance
        benchmark_value = np.sum(self.equal_weight_shares * self.price_array[self.time])
        benchmark_return = (benchmark_value - self.last_benchmark_value) / self.last_benchmark_value
        self.last_benchmark_value = benchmark_value

        # Calculate agent's return
        agent_return = (next_total_asset - self.total_asset) / self.total_asset
        
        # Update return history and calculate volatility
        self.return_history.append(agent_return)
        if len(self.return_history) > self.volatility_window:
            self.return_history.pop(0)
        self.volatility = np.std(self.return_history) if len(self.return_history) > 1 else 0.0

        # Reward components
        base_reward = agent_return * 100  # Scale up returns
        relative_reward = (agent_return - benchmark_return) * 50  # Bonus/penalty vs benchmark
        risk_penalty = -max(0, self.volatility - 0.02) * 20  # Penalize excess volatility

        # Combined reward
        reward = base_reward + relative_reward + risk_penalty
        
        self.total_asset = next_total_asset
        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash
        
        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash
            
        info = {
            'total_asset': float(self.total_asset),
            'cash': float(self.cash),
            'stocks': self.stocks.copy(),
            'gamma_return': float(self.gamma_return),
            'episode_return': float(getattr(self, 'episode_return', 0.0))
        }
        
        return state, reward, done, False, info

    def get_state(self):
        state_list = []
        state_list.append(self.cash * 2 ** -18)
        state_list.extend([0.0, 0.0])
        
        state_list.extend(self.stocks.tolist())
        state_list.extend(self.current_price.tolist())
        holdings = self.stocks * self.current_price
        state_list.extend(holdings.tolist())
        state_list.extend(self.current_tech.tolist())
        
        state = np.array(state_list, dtype=np.float32).copy()
        assert len(state) == self.state_dim, f"State dimension mismatch. Expected {self.state_dim}, got {len(state)}"
        return state.reshape(-1)
    
    def close(self):
        pass

    def _generate_action_normalizer(self):
        action_norm_vector = []
        price_0 = self.price_array[0]
        for price in price_0:
            x = math.floor(math.log(price, 10))
            action_norm_vector.append(1/((10)**x)) 
            
        action_norm_vector = np.asarray(action_norm_vector) * 10000
        self.action_norm_vector = np.asarray(action_norm_vector)
