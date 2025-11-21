import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


CSV_FILE = "Metrics/RL/rl_training_log.csv"  
OUTPUT_DIR = "Metrics/RL/rl_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_loss_curves(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Please save your data as 'rl_training_log.csv'")
        return

    try:
        df = pd.read_csv(filename, header=None, names=["Timestamp", "Actor_Loss", "Critic_Loss"])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Normalize Time to start at 0
    df['Time'] = df['Timestamp'] - df['Timestamp'].iloc[0]

    # Create Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Actor Loss (Primary Axis - Left)
    color = 'tab:blue'
    ax1.set_xlabel('Training Time (seconds)')
    ax1.set_ylabel('Actor Loss', color=color)
    # Use rolling mean to smooth out the noise typical in RL training
    ax1.plot(df['Time'], df['Actor_Loss'].rolling(window=10).mean(), color=color, label='Actor Loss (Smoothed)', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create a second y-axis for Critic Loss (Right)
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Critic Loss (MSE)', color=color)
    ax2.plot(df['Time'], df['Critic_Loss'].rolling(window=10).mean(), color=color, linestyle='--', label='Critic Loss (Smoothed)', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title & Layout
    plt.title("RL Training Convergence: Actor vs. Critic Loss")
    fig.tight_layout()  
    
    save_path = f"{OUTPUT_DIR}/graph_rl_loss_convergence.png"
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_loss_curves(CSV_FILE)