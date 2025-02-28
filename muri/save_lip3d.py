import numpy as np
import os
import pickle
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm 

# Import from our LIP3D module
from LIP3D import LinearInvertedPendulum3D, impact_event

def simulate_trajectory(model, initial_state, r0, t_max=5.0, dt=0.01):
    """
    Simulate the complete 3D LIP trajectory from t=0 to t=t_max, handling resets automatically
    
    Parameters:
    -----------
    model : LinearInvertedPendulum3D
        The LIP model
    initial_state : array-like
        Initial state [x, y, x_dot, y_dot]
    r0 : float
        Radius of the impact boundary
    t_max : float, optional
        Maximum simulation time (default: 5.0s)
    dt : float, optional
        Time step for output trajectory (default: 0.01s)
        
    Returns:
    --------
    tuple
        (times, states, reset_times)
        - times: array of time points
        - states: array of states at each time point
        - reset_times: list of times when resets occurred
    """
    # Initialize arrays to store the trajectory
    times = []
    states = []
    reset_times = []
    
    # Start with the initial state
    current_time = 0.0
    current_state = np.array(initial_state)
    
    # Add initial state to trajectory
    times.append(current_time)
    states.append(current_state.copy())
    
    # Simulate until t_max
    while current_time < t_max:
        # Determine how much time is left
        time_left = t_max - current_time
        
        # Simulate until next impact or until t_max
        sol = solve_ivp(
            model.dynamics, 
            (0, time_left), 
            current_state, 
            method='RK45', 
            events=impact_event(r0, min_time=0.05),
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.1
        )
        
        # Check if we hit an impact
        impact_detected = len(sol.t_events[0]) > 0
        
        # Generate dense output for smooth trajectory
        t_dense = np.linspace(0, sol.t[-1], max(int(sol.t[-1] / dt), 2))
        states_dense = sol.sol(t_dense)
        
        # Add time offset for continuous timing
        t_dense += current_time
        
        # Add this segment to the trajectory (skip the first point if not the first segment)
        if len(times) > 1:
            times.extend(t_dense[1:])  # Skip first point to avoid duplicates
            states.extend(states_dense[:, 1:].T)  # Transpose to get [n_points, 4] shape
        else:
            times.extend(t_dense)
            states.extend(states_dense.T)
        
        # Update current time
        current_time += sol.t[-1]
        
        if impact_detected:
            # Record the reset time
            reset_times.append(current_time)
            
            # Get the state at impact
            final_state = sol.y[:, -1]
            
            # Apply reset map
            current_state = model.reset_map(final_state)
            
            # Add the reset state to the trajectory
            times.append(current_time)
            states.append(current_state.copy())
        else:
            # We reached t_max without impact
            break
    
    # Convert lists to numpy arrays
    times = np.array(times)
    states = np.array(states)
    
    return times, states, reset_times

def is_flowing_inward(x0, y0, vx, vy):
    """
    Check if the velocity vector points inward toward the circle
    
    Parameters:
    -----------
    x0, y0 : float
        Initial position
    vx, vy : float
        Initial velocity
    
    Returns:
    --------
    bool
        True if the trajectory flows inward, False otherwise
    """
    # Check if the dot product of position and velocity is negative
    # This means the velocity vector points inward
    dot_product = x0 * vx + y0 * vy
    return dot_product < 0

def generate_and_save_trajectories(num_samples=1000, save_dir='lip3d_trajectories'):
    """
    Generate and save trajectories with initial conditions on the upper half of the unit circle
    
    Parameters:
    -----------
    num_samples : int, optional
        Number of trajectories to generate (default: 1000)
    save_dir : str, optional
        Directory to save the trajectories (default: 'lip3d_trajectories')
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Base parameters
    z0 = 1.0  # Height of the constraint plane
    r0 = 1.0  # Radius of the impact boundary
    t_max = 5.0  # Maximum simulation time
    
    # Create a model (we'll update x0, y0 for each trajectory)
    lip_model = LinearInvertedPendulum3D(0, 0, z0)
    
    # Store metadata for all trajectories
    metadata = {
        'num_samples': num_samples,
        'r0': r0,
        'z0': z0,
        't_max': t_max,
        'trajectories': []
    }
    
    # Generate trajectories
    count = 0
    attempts = 0
    max_attempts = num_samples * 10  # Limit the number of attempts
    
    with tqdm(total=num_samples) as pbar:
        while count < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Generate random angle in the upper half of the circle
            theta = np.random.uniform(0, np.pi)
            
            # Calculate position on the circle
            x0 = r0 * np.cos(theta)
            y0 = r0 * np.sin(theta)
            
            # Update model with new foothold position
            lip_model.x0 = x0
            lip_model.y0 = y0
            
            # Generate random velocities
            vx = np.random.uniform(-5, 5)
            vy = np.random.uniform(-5, 5)
            
            # Check if the trajectory flows inward
            if is_flowing_inward(x0, y0, vx, vy):
                # Create initial state (position on the circle, with the given velocity)
                initial_state = np.array([x0, y0, vx, vy])
                
                # Simulate trajectory
                times, states, reset_times = simulate_trajectory(lip_model, initial_state, r0, t_max)
                
                # Only save if we have a valid trajectory with at least one reset
                if len(reset_times) > 0:
                    # Save trajectory data
                    trajectory_data = {
                        'id': count,
                        'initial_state': initial_state,
                        'theta': theta,
                        'times': times,
                        'states': states,
                        'reset_times': reset_times,
                        'sync_measure_initial': lip_model.synchronization_measure(initial_state)
                    }
                    
                    # Save to file
                    with open(os.path.join(save_dir, f'trajectory_{count:04d}.pkl'), 'wb') as f:
                        pickle.dump(trajectory_data, f)
                    
                    # Add metadata
                    metadata['trajectories'].append({
                        'id': count,
                        'initial_state': initial_state.tolist(),
                        'theta': theta,
                        'num_resets': len(reset_times),
                        'duration': times[-1]
                    })
                    
                    count += 1
                    pbar.update(1)
                    
                    # Print progress every 100 trajectories
                    if count % 100 == 0:
                        print(f"Generated {count} trajectories ({attempts} attempts)")
    
    # Save metadata
    with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Generated {count} trajectories in {attempts} attempts")
    print(f"Saved to {os.path.abspath(save_dir)}")
    
    # Create a visualization of the initial conditions
    visualize_initial_conditions(metadata, save_dir)

def visualize_initial_conditions(metadata, save_dir):
    """
    Create a visualization of the initial conditions
    
    Parameters:
    -----------
    metadata : dict
        Metadata for all trajectories
    save_dir : str
        Directory to save the visualization
    """
    plt.figure(figsize=(10, 10))
    
    # Draw the unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
    
    # Extract initial states
    initial_states = np.array([traj['initial_state'] for traj in metadata['trajectories']])
    
    # Plot initial positions
    plt.scatter(initial_states[:, 0], initial_states[:, 1], c='b', alpha=0.5, label='Initial positions')
    
    # Plot velocity vectors (scaled down for visibility)
    scale = 0.1
    for state in initial_states:
        x, y, vx, vy = state
        plt.arrow(x, y, scale*vx, scale*vy, head_width=0.03, head_length=0.05, fc='r', ec='r', alpha=0.3)
    
    plt.grid(True)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Initial Conditions for LIP3D Trajectories')
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'initial_conditions.png'), dpi=300)
    plt.close()

def main():
    """
    Main function to run the trajectory generation with a small test first
    """
    # First run a small test with 5 samples
    print("Running test with 5 samples...")
    generate_and_save_trajectories(num_samples=5, save_dir='lip3d_test')
    
    # Ask user if they want to proceed with the full 1000 samples
    response = input("\nTest complete. Proceed with generating 1000 samples? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\nGenerating 1000 samples...")
        generate_and_save_trajectories(num_samples=1000, save_dir='lip3d_trajectories')
    else:
        print("\nFull generation skipped.")

if __name__ == "__main__":
    main() 