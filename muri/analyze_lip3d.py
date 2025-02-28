import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_all_trajectories(data_dir='lip3d_trajectories'):
    """
    Load all trajectory data from pickle files and extract states
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the trajectory pickle files
    
    Returns:
    --------
    dict
        Dictionary containing all trajectory states and metadata
    """
    # First load the metadata
    metadata_path = os.path.join(data_dir, 'metadata.pkl')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Found metadata for {len(metadata['trajectories'])} trajectories")
    
    # Initialize storage for all trajectories
    all_data = {
        'metadata': metadata,
        'trajectories': []
    }
    
    # Load each trajectory file
    print("Loading trajectory files...")
    for traj_info in tqdm(metadata['trajectories']):
        traj_id = traj_info['id']
        traj_path = os.path.join(data_dir, f'trajectory_{traj_id:04d}.pkl')
        
        if not os.path.exists(traj_path):
            print(f"Warning: Trajectory file {traj_path} not found, skipping")
            continue
        
        with open(traj_path, 'rb') as f:
            traj_data = pickle.load(f)
        
        # Extract just the states and reset information
        trajectory = {
            'id': traj_id,
            'initial_state': traj_data['initial_state'],
            'states': traj_data['states'],
            'reset_indices': [np.where(traj_data['times'] == rt)[0][0] for rt in traj_data['reset_times']],
            'num_resets': len(traj_data['reset_times'])
        }
        
        all_data['trajectories'].append(trajectory)
    
    print(f"Successfully loaded {len(all_data['trajectories'])} trajectories")
    return all_data

def analyze_trajectories(all_data):
    """
    Perform basic analysis on the trajectory data
    
    Parameters:
    -----------
    all_data : dict
        Dictionary containing all trajectory states and metadata
    """
    # Get some basic statistics
    num_trajectories = len(all_data['trajectories'])
    total_states = sum(len(traj['states']) for traj in all_data['trajectories'])
    avg_states_per_traj = total_states / num_trajectories
    
    reset_counts = [traj['num_resets'] for traj in all_data['trajectories']]
    
    print(f"Total trajectories: {num_trajectories}")
    print(f"Total state points: {total_states}")
    print(f"Average points per trajectory: {avg_states_per_traj:.2f}")
    print(f"Average resets per trajectory: {np.mean(reset_counts):.2f}")
    print(f"Min resets: {min(reset_counts)}, Max resets: {max(reset_counts)}")
    
    # Create a histogram of reset counts
    plt.figure(figsize=(10, 6))
    plt.hist(reset_counts, bins=range(min(reset_counts), max(reset_counts)+2), alpha=0.7)
    plt.xlabel('Number of Resets')
    plt.ylabel('Count')
    plt.title('Distribution of Reset Counts per Trajectory')
    plt.grid(True, alpha=0.3)
    plt.savefig('reset_distribution.png', dpi=300)
    plt.close()
    
    # Create a scatter plot of initial positions colored by number of resets
    plt.figure(figsize=(10, 10))
    
    # Draw the unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
    
    # Extract initial states and reset counts
    initial_x = [traj['initial_state'][0] for traj in all_data['trajectories']]
    initial_y = [traj['initial_state'][1] for traj in all_data['trajectories']]
    
    # Create scatter plot
    scatter = plt.scatter(initial_x, initial_y, c=reset_counts, cmap='viridis', 
                         alpha=0.7, s=30, edgecolors='k', linewidths=0.5)
    
    plt.colorbar(scatter, label='Number of Resets')
    plt.grid(True)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Initial Positions Colored by Number of Resets')
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.savefig('initial_positions_by_resets.png', dpi=300)
    plt.close()

def save_combined_states(all_data, output_file='all_lip3d_states.npz'):
    """
    Extract and save all states from all trajectories into a single file
    
    Parameters:
    -----------
    all_data : dict
        Dictionary containing all trajectory states and metadata
    output_file : str
        Path to save the combined states
    """
    # Extract all states and create a mapping to original trajectories
    all_states = []
    traj_indices = []
    reset_flags = []
    
    for i, traj in enumerate(all_data['trajectories']):
        states = traj['states']
        reset_idx = traj['reset_indices']
        
        # Create a reset flag array (1 at reset points, 0 elsewhere)
        flags = np.zeros(len(states))
        flags[reset_idx] = 1
        
        all_states.append(states)
        traj_indices.extend([i] * len(states))
        reset_flags.extend(flags)
    
    # Combine all states into a single array
    combined_states = np.vstack(all_states)
    traj_indices = np.array(traj_indices)
    reset_flags = np.array(reset_flags)
    
    print(f"Combined state array shape: {combined_states.shape}")
    print(f"Trajectory indices array shape: {traj_indices.shape}")
    print(f"Reset flags array shape: {reset_flags.shape}")
    
    # Save to a compressed numpy file
    np.savez_compressed(
        output_file,
        states=combined_states,
        traj_indices=traj_indices,
        reset_flags=reset_flags
    )
    
    print(f"Saved combined states to {output_file}")

def main():
    """
    Main function to load and analyze trajectory data
    """
    # Ask for the data directory
    data_dir = input("Enter the directory containing trajectory data [lip3d_trajectories]: ") or "lip3d_trajectories"
    
    # Load all trajectory data
    all_data = load_all_trajectories(data_dir)
    
    # Analyze the trajectories
    analyze_trajectories(all_data)
    
    # Save combined states
    save_combined_states(all_data)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 