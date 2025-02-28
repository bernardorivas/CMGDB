import CMGDB
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

# Add the muri directory to the path to import LIP3D
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'muri'))
from LIP3D import LinearInvertedPendulum3D, impact_event

def main():
    """
    Main function to demonstrate CMGDB with the 3D Linear Inverted Pendulum hybrid system.
    """
    print("Hybrid System Example: 3D Linear Inverted Pendulum")
    print("=================================================")
    
    # Define the LIP3D model parameters
    z0 = 1.0  # Height of the constraint plane
    r0 = 1.0  # Radius of the impact boundary
    
    # Generate data for the hybrid system
    print("\nGenerating data from multiple trajectories...")
    X, Y = generate_lip3d_data(z0=z0, r0=r0, num_samples=1000)
    
    # Plot a sample of the data
    plot_sample_data(X, Y, r0)
    
    # Create a box map from the data
    print("\nCreating box map from hybrid system data...")
    F = CMGDB.BoxMapData(X, Y, map_empty='interpolate')
    
    # Define the domain and subdivision parameters
    # We'll focus on the x-y plane for simplicity
    lower_bounds = [-1.0, -1.0]  # [min_x, min_y]
    upper_bounds = [1.0, 1.0]    # [max_x, max_y]
    
    subdiv_min = 6
    subdiv_max = 8
    subdiv_init = 4
    subdiv_limit = 10000
    
    # Create the model
    model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit, 
                        lower_bounds, upper_bounds, F)
    
    # Compute the Morse graph
    print("\nComputing Morse graph...")
    start_time = time.time()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    end_time = time.time()
    print(f"Morse graph computation completed in {end_time - start_time:.2f} seconds")
    
    # Visualize the Morse graph
    print("\nVisualizing Morse graph...")
    CMGDB.PlotMorseGraph(morse_graph)
    plt.savefig("lip3d_morse_graph.png")
    plt.close()
    
    # Visualize the Morse sets in phase space
    print("Visualizing Morse sets in phase space...")
    CMGDB.PlotMorseSets(morse_graph, fig_w=8, fig_h=8)
    plt.savefig("lip3d_morse_sets.png")
    plt.close()
    
    # Compare with a different radius
    print("\nComparing with a different impact boundary radius...")
    compare_different_radius(r0=0.8, z0=z0, 
                            subdiv_min=subdiv_min, subdiv_max=subdiv_max,
                            subdiv_init=subdiv_init, subdiv_limit=subdiv_limit,
                            lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    
    print("\nAnalysis complete. Results saved as PNG files.")

def simulate_lip3d_step(model, initial_state, r0, dt=0.01, max_time=5.0):
    """
    Simulate a single step of the LIP3D model until impact or max_time.
    
    Args:
        model: LinearInvertedPendulum3D model
        initial_state: Initial state [x, y, x_dot, y_dot]
        r0: Radius of the impact boundary
        dt: Time step for integration
        max_time: Maximum simulation time
        
    Returns:
        tuple: (states, next_states, had_impact)
            - states: List of states before impact
            - next_states: List of states after one time step
            - had_impact: Boolean indicating if impact occurred
    """
    # Initialize lists to store states and next states
    states = []
    next_states = []
    
    # Current state and time
    state = np.array(initial_state)
    t = 0.0
    had_impact = False
    
    while t < max_time:
        # Store current state
        states.append(state.copy())
        
        # Compute state derivative
        dstate = model.dynamics(t, state)
        
        # Euler integration for one time step
        next_state = state + dt * np.array(dstate)
        
        # Check if we hit the impact boundary
        x, y = next_state[0], next_state[1]
        radius = np.sqrt(x**2 + y**2)
        
        if radius >= r0:
            # Apply reset map
            reset_state = model.reset_map(next_state)
            next_states.append(reset_state)
            had_impact = True
            break
        else:
            # Store next state and continue
            next_states.append(next_state)
            state = next_state
            t += dt
    
    return states, next_states, had_impact

def generate_lip3d_data(z0=1.0, r0=1.0, num_samples=1000):
    """
    Generate data for the LIP3D system by sampling initial conditions.
    
    Args:
        z0: Height of the constraint plane
        r0: Radius of the impact boundary
        num_samples: Number of initial conditions to sample
        
    Returns:
        X: Array of states
        Y: Array of next states
    """
    # Parameters for simulation
    dt = 0.1  # Larger time step for discrete transitions
    max_time = 5.0
    
    # Lists to store all states and next states
    all_states = []
    all_next_states = []
    
    # Counter for successful trajectories (those with impacts)
    successful = 0
    
    print(f"Generating {num_samples} samples...")
    
    while successful < num_samples:
        # Sample random initial conditions
        theta = np.random.uniform(0, 2*np.pi)
        if abs(theta - np.pi) < 0.1:  # Avoid values too close to π
            continue
            
        # Random radius between 0.1*r0 and 0.9*r0 (inside the boundary)
        radius = np.random.uniform(0.1*r0, 0.9*r0)
        
        # Calculate position
        x0 = radius * np.cos(theta)
        y0 = radius * np.sin(theta)
        
        # Random velocities
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 1)
        
        # Create initial state
        initial_state = np.array([x0, y0, vx, vy])
        
        # Create LIP3D model with foothold at origin
        model = LinearInvertedPendulum3D(0, 0, z0)
        
        # Simulate one step
        states, next_states, had_impact = simulate_lip3d_step(model, initial_state, r0, dt, max_time)
        
        # Only use trajectories that had an impact
        if had_impact and len(states) > 0:
            all_states.extend(states)
            all_next_states.extend(next_states[:len(states)])
            successful += 1
            
            if successful % 100 == 0:
                print(f"  Generated {successful} successful trajectories")
    
    # Convert to numpy arrays
    X = np.array(all_states)
    Y = np.array(all_next_states)
    
    # For CMGDB, we'll only use the position components (x, y)
    X = X[:, :2]
    Y = Y[:, :2]
    
    print(f"Generated {len(X)} state-to-next-state pairs from {successful} trajectories")
    return X, Y

def plot_sample_data(X, Y, r0):
    """
    Plot a sample of the data points.
    
    Args:
        X: Array of states
        Y: Array of next states
        r0: Radius of the impact boundary
    """
    plt.figure(figsize=(10, 10))
    
    # Draw the impact boundary circle
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(r0*np.cos(theta), r0*np.sin(theta), 'k--', alpha=0.5, label='Impact boundary')
    
    # Sample a subset of points for clarity
    sample_size = min(1000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    
    # Plot the sampled points
    plt.scatter(X[indices, 0], X[indices, 1], c='blue', s=10, alpha=0.5, label='States')
    plt.scatter(Y[indices, 0], Y[indices, 1], c='red', s=10, alpha=0.5, label='Next States')
    
    # Draw arrows for a smaller subset
    arrow_indices = np.random.choice(indices, 100, replace=False)
    for i in arrow_indices:
        plt.arrow(X[i, 0], X[i, 1], Y[i, 0] - X[i, 0], Y[i, 1] - X[i, 1], 
                  head_width=0.02, head_length=0.03, fc='k', ec='k', alpha=0.3)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('LIP3D: State Transitions in X-Y Plane')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig("lip3d_data.png")
    plt.close()
    
    print("Sample data plot saved as 'lip3d_data.png'")

def compare_different_radius(r0, z0, subdiv_min, subdiv_max, subdiv_init, subdiv_limit, 
                            lower_bounds, upper_bounds):
    """
    Compare the dynamics with a different impact boundary radius.
    
    Args:
        r0: New impact boundary radius
        z0: Height of the constraint plane
        subdiv_min, subdiv_max, subdiv_init, subdiv_limit: Subdivision parameters
        lower_bounds, upper_bounds: Domain bounds
    """
    # Generate data with the new radius
    X_new, Y_new = generate_lip3d_data(z0=z0, r0=r0, num_samples=1000)
    
    # Plot sample data
    plot_sample_data(X_new, Y_new, r0)
    
    # Create a new box map
    F_new = CMGDB.BoxMapData(X_new, Y_new, map_empty='interpolate')
    
    # Create a new model
    model_new = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit, 
                           lower_bounds, upper_bounds, F_new)
    
    # Compute the new Morse graph
    print(f"Computing Morse graph for r0={r0}...")
    start_time = time.time()
    morse_graph_new, map_graph_new = CMGDB.ComputeMorseGraph(model_new)
    end_time = time.time()
    print(f"Morse graph computation completed in {end_time - start_time:.2f} seconds")
    
    # Visualize the new Morse graph
    print(f"Visualizing Morse graph for r0={r0}...")
    CMGDB.PlotMorseGraph(morse_graph_new)
    plt.savefig(f"lip3d_morse_graph_r0_{r0}.png")
    plt.close()
    
    # Visualize the new Morse sets in phase space
    print(f"Visualizing Morse sets in phase space for r0={r0}...")
    CMGDB.PlotMorseSets(morse_graph_new, fig_w=8, fig_h=8)
    plt.savefig(f"lip3d_morse_sets_r0_{r0}.png")
    plt.close()

if __name__ == "__main__":
    main()