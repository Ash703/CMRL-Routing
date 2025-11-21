import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


CSV_FILE = "Metrics/RL/rl_actions_log.csv"  
OUTPUT_DIR = "Metrics/RL/rl_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None
    cols = ["Timestamp", "Source", "Destination", "FlowType", "Spine1_Prob", "Spine2_Prob", "Spine3_Prob"]
    
    try:

        df = pd.read_csv(filename, header=None, names=cols)
        
        # Normalize Timestamp to start at 0
        df['Time'] = df['Timestamp'] - df['Timestamp'].iloc[0]
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def plot_probability_evolution(df):
    """
    Plots the probability weights for the 3 spines over time.
    We filter for a specific flow type (e.g., Elephant) to see decision stability.
    """
    elephants = df[df['FlowType'] == 'elephant']
    
    if elephants.empty:
        print("No Elephant flows found, plotting all flows...")
        plot_data = df.iloc[::10] # Downsample for readability if too large
    else:
        plot_data = elephants

    plt.figure(figsize=(12, 6))
    
    # Plot stacked area or lines
    plt.plot(plot_data['Time'], plot_data['Spine1_Prob'], label='Spine 1', alpha=0.8, linewidth=1.5)
    plt.plot(plot_data['Time'], plot_data['Spine2_Prob'], label='Spine 2', alpha=0.8, linewidth=1.5)
    plt.plot(plot_data['Time'], plot_data['Spine3_Prob'], label='Spine 3', alpha=0.8, linewidth=1.5)
    
    plt.title("Evolution of Routing Probabilities Over Time")
    plt.xlabel("Experiment Time (s)")
    plt.ylabel("Probability Weight")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{OUTPUT_DIR}/graph_rl_probability_evolution.png")
    print(f"Saved: {OUTPUT_DIR}/graph_rl_probability_evolution.png")
    plt.close()

def plot_policy_distribution(df):
    """
    Box plot showing the distribution of probabilities for each Spine.
    This reveals if the agent favors one spine significantly.
    """
    plt.figure(figsize=(8, 6))
    
    data = [df['Spine1_Prob'], df['Spine2_Prob'], df['Spine3_Prob']]
    
    plt.boxplot(data, labels=['Spine 1', 'Spine 2', 'Spine 3'], patch_artist=True)
    
    plt.title("Agent Policy Distribution (Path Preference)")
    plt.ylabel("Probability Assigned")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(f"{OUTPUT_DIR}/graph_rl_policy_boxplot.png")
    print(f"Saved: {OUTPUT_DIR}/graph_rl_policy_boxplot.png")
    plt.close()

def plot_confidence(df):
    """
    Plots the 'Confidence' (Max Probability) over time.
    If Max Prob -> 1.0, the agent is certain. If -> 0.33, it is guessing.
    """
    df['Confidence'] = df[['Spine1_Prob', 'Spine2_Prob', 'Spine3_Prob']].max(axis=1)
    
    plt.figure(figsize=(12, 5))
    
    # Moving average for smooth trend line
    window = max(1, len(df) // 50)
    smoothed = df['Confidence'].rolling(window=window).mean()
    
    plt.scatter(df['Time'], df['Confidence'], alpha=0.1, color='gray', s=5, label='Raw Decisions')
    plt.plot(df['Time'], smoothed, color='red', linewidth=2, label='Trend (Moving Avg)')
    
    plt.title("Agent Confidence (Convergence Check)")
    plt.xlabel("Time (s)")
    plt.ylabel("Max Probability (Certainty)")
    plt.ylim(0.3, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{OUTPUT_DIR}/graph_rl_convergence.png")
    print(f"Saved: {OUTPUT_DIR}/graph_rl_convergence.png")
    plt.close()

if __name__ == "__main__":
    df = load_data(CSV_FILE)
    
    if df is not None and not df.empty:
        print(f"Loaded {len(df)} training steps.")
        plot_probability_evolution(df)
        plot_policy_distribution(df)
        plot_confidence(df)
    else:
        print("Could not generate plots. Check CSV file.")