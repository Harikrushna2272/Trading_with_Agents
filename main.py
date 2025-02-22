import yfinance as yf
import pandas as pd
from src.environment import TradingEnvironment
from src.agent import DQNAgent

def prepare_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def main():
    # Parameters
    symbol = "HDFC"
    start_date = "2000-01-01"
    end_date = "2021-02-14"
    
    # Prepare data
    data = prepare_data(symbol, start_date, end_date)
    
    # Initialize environment and agent
    env = TradingEnvironment(data)
    agent = DQNAgent(state_size=4, action_size=3)
    
    # Training parameters
    batch_size = 32
    episodes = 500
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        agent.replay(batch_size)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
    
    print("Training Complete!")
    
    # Test the agent
    test_env = TradingEnvironment(data)
    state = test_env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = test_env.step(action)
        state = next_state if next_state is not None else state
    
    final_balance = test_env.balance
    profit = final_balance - test_env.initial_balance
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Profit: ${profit:.2f}")

if __name__ == "__main__":
    main()
