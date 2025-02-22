import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from src.environment import TradingEnvironment
from src.models import DQN, DQNAgent

def load_and_preprocess_data():
    """Load and preprocess HDFC data from local CSV"""
    try:
        # Load data from CSV file in data folder
        data = pd.read_csv('data/HDFC.csv')
        
        # Ensure the date column is properly formatted
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Calculate technical indicators
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['Returns'] = data['Close'].pct_change()
        
        # Drop any missing values
        data.dropna(inplace=True)
        
        # Reset index to make sure we have consecutive integers as index
        data.reset_index(inplace=True)
        
        return data
    
    except FileNotFoundError:
        print("Error: HDFC.csv not found in data folder")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def get_state(data, index):
    """Extract state from data at given index"""
    try:
        return np.array([
            float(data.loc[index, 'Close']),
            float(data.loc[index, 'SMA_5']),
            float(data.loc[index, 'SMA_20']),
            float(data.loc[index, 'Returns'])
        ])
    except KeyError as e:
        print(f"Error accessing data at index {index}: {str(e)}")
        return None

def main():
    # Load and preprocess data
    data = load_and_preprocess_data()
    if data is None:
        return
    
    print("Data loaded successfully. Shape:", data.shape)
    
    # Initialize environment and agent
    try:
        env = TradingEnvironment(data)
        state_size = 4  # Close, SMA_5, SMA_20, Returns
        action_size = 3  # Buy, Sell, Hold
        agent = DQNAgent(state_size, action_size)
        
        # Training parameters
        batch_size = 32
        episodes = 500
        
        # Training loop
        total_rewards = []
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                if next_state is not None:
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                
            agent.replay(batch_size)
            total_rewards.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}")
        
        # Final evaluation
        print("\nTraining Complete!")
        print(f"Final Balance: ${env.balance:.2f}")
        print(f"Total Profit/Loss: ${(env.balance - env.initial_balance):.2f}")
        
        # Save the trained model
        torch.save(agent.model.state_dict(), 'models/trading_model.pth')
        print("Model saved to models/trading_model.pth")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()
