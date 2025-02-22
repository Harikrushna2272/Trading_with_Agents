import numpy as np

class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0
        self.max_index = len(data) - 1
    
    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0
        return self.get_state()
    
    def get_state(self):
        if self.index > self.max_index:
            return None
        try:
            return np.array([
                float(self.data.loc[self.index, 'Close']),
                float(self.data.loc[self.index, 'SMA_5']),
                float(self.data.loc[self.index, 'SMA_20']),
                float(self.data.loc[self.index, 'Returns'])
            ])
        except Exception as e:
            print(f"Error getting state at index {self.index}: {str(e)}")
            return None
    
    def step(self, action):
        if self.index >= self.max_index:
            return None, 0, True, {}
        
        current_price = float(self.data.loc[self.index, 'Close'])
        reward = 0
        
        try:
            if action == 1:  # Buy
                shares_to_buy = self.balance // current_price
                if shares_to_buy > 0:
                    self.holdings += shares_to_buy
                    self.balance -= shares_to_buy * current_price
            
            elif action == 2:  # Sell
                if self.holdings > 0:
                    self.balance += self.holdings * current_price
                    self.holdings = 0
            
            self.index += 1
            done = self.index >= self.max_index
            
            if done:
                # Calculate final portfolio value
                final_value = self.balance + (self.holdings * current_price)
                reward = final_value - self.initial_balance
            
            next_state = self.get_state()
            return next_state, reward, done, {}
            
        except Exception as e:
            print(f"Error during step: {str(e)}")
            return None, 0, True, {}
