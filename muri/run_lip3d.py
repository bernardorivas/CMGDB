import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Import from our LIP3D module
from LIP3D import LinearInvertedPendulum3D, impact_event

"""
# Commented out as we're using simulate_trajectory instead
def simulate_multiple_steps(model, initial_state, r0, num_steps=10, max_time_per_step=50.0):
    # Function implementation commented out
    pass
"""

def simulate_trajectory(model, initial_state, r0, t_max, dt=0.01):
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
    t_max : float
        Maximum simulation time
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
    
    # Check if initial state is already outside the boundary
    x0, y0 = current_state[0], current_state[1]
    initial_radius = np.sqrt(x0**2 + y0**2)
    if initial_radius >= r0:
        print(f"Warning: Initial state is already at/outside the boundary (r={initial_radius:.6f}, r0={r0:.6f})")
    
    # Simulate until t_max
    step_count = 0
    while current_time < t_max:
        step_count += 1
        print(f"Step {step_count}: t={current_time:.6f}, state={current_state}")
        
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
            rtol=1e-8,  # Tighter tolerance
            atol=1e-10,  # Tighter tolerance
            max_step=0.1  # Limit maximum step size
        )
        
        # Check if we hit an impact
        impact_detected = len(sol.t_events[0]) > 0
        
        # Generate dense output for smooth trajectory
        t_dense = np.linspace(0, sol.t[-1], max(int(sol.t[-1] / dt), 2))
        states_dense = sol.sol(t_dense)
        
        # Verify impact detection by checking final state
        final_x, final_y = sol.y[0, -1], sol.y[1, -1]
        final_radius = np.sqrt(final_x**2 + final_y**2)
        
        # Debug output
        print(f"  Integration ended at t={sol.t[-1]:.6f}, radius={final_radius:.6f}, r0={r0:.6f}")
        print(f"  Impact detected by event: {impact_detected}")
        
        # Double-check if we're at the boundary
        at_boundary = abs(final_radius - r0) < 1e-6
        if at_boundary and not impact_detected:
            print(f"  Warning: At boundary but impact not detected by event function")
            impact_detected = True
        
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
            
            print(f"  Reset at t={current_time:.3f}s: {final_state} -> {current_state}")
            
            # Verify that the reset state is inside the boundary
            reset_x, reset_y = current_state[0], current_state[1]
            reset_radius = np.sqrt(reset_x**2 + reset_y**2)
            if reset_radius >= r0:
                print(f"  Warning: Reset state is outside the boundary (r={reset_radius:.6f}, r0={r0:.6f})")
        else:
            # We reached t_max without impact
            print(f"  Reached t_max={t_max} without further impacts")
            break
    
    # Convert lists to numpy arrays
    times = np.array(times)
    states = np.array(states)
    
    return times, states, reset_times

def plot_trajectory(times, states, reset_times, r0, model):
    """
    Plot the complete trajectory with resets
    
    Parameters:
    -----------
    times : array-like
        Array of time points
    states : array-like
        Array of states at each time point
    reset_times : list
        List of times when resets occurred
    r0 : float
        Radius of the impact boundary
    model : LinearInvertedPendulum3D
        The LIP model for calculating metrics
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Position x vs position y plot (subplot 1)
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Draw the circular impact boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(r0*np.cos(theta), r0*np.sin(theta), 'k--', alpha=0.5, label='Impact boundary')
    
    # Find indices of reset points
    reset_indices = []
    for reset_time in reset_times:
        idx = np.where(times == reset_time)[0][0]
        reset_indices.append(idx)
    
    # Add start and end indices to create segments
    segment_indices = [0] + reset_indices + [len(times)-1]
    
    # Plot each continuous segment separately
    for i in range(len(segment_indices)-1):
        start_idx = segment_indices[i]
        end_idx = segment_indices[i+1]
        
        # Plot continuous segment
        ax1.plot(states[start_idx:end_idx+1, 0], states[start_idx:end_idx+1, 1], 
                 'b-', linewidth=1.5, label='Trajectory' if i==0 else "")
        
        # If this is a reset point (not the end), draw the reset transition as a dashed line
        if i < len(reset_indices):
            reset_idx = reset_indices[i]
            # The reset transition connects states[reset_idx] to states[reset_idx+1]
            if reset_idx+1 < len(states):
                ax1.plot([states[reset_idx, 0], states[reset_idx+1, 0]], 
                         [states[reset_idx, 1], states[reset_idx+1, 1]], 
                         'r--', linewidth=1.5, alpha=0.7, label='Reset transition' if i==0 else "")
    
    # Mark start and end points
    ax1.plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
    ax1.plot(states[-1, 0], states[-1, 1], 'ro', markersize=8, label='End')
    
    # Mark reset points
    for i, reset_idx in enumerate(reset_indices):
        ax1.plot(states[reset_idx, 0], states[reset_idx, 1], 'k*', markersize=10, 
                label='Reset point' if i==0 else "")
    
    ax1.grid(True)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('X-Y Trajectory')
    ax1.set_aspect('equal')  # Equal aspect ratio
    ax1.legend()
    
    # Velocity plot (subplot 2)
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Plot each continuous segment separately
    for i in range(len(segment_indices)-1):
        start_idx = segment_indices[i]
        end_idx = segment_indices[i+1]
        
        # Plot continuous segments
        ax2.plot(times[start_idx:end_idx+1], states[start_idx:end_idx+1, 2], 
                'b-', label='ẋ' if i==0 else "")
        ax2.plot(times[start_idx:end_idx+1], states[start_idx:end_idx+1, 3], 
                'r-', label='ẏ' if i==0 else "")
    
    # Mark reset points on velocity plot
    for reset_time in reset_times:
        ax2.axvline(x=reset_time, color='k', linestyle='--', alpha=0.5)
    
    ax2.grid(True)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs Time')
    ax2.legend()
    
    # Phase space (subplot 3)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Plot each continuous segment separately
    for i in range(len(segment_indices)-1):
        start_idx = segment_indices[i]
        end_idx = segment_indices[i+1]
        
        # Plot continuous segments
        ax3.plot(states[start_idx:end_idx+1, 0], states[start_idx:end_idx+1, 2], 
                'b-', label='X phase' if i==0 else "")
        ax3.plot(states[start_idx:end_idx+1, 1], states[start_idx:end_idx+1, 3], 
                'r-', label='Y phase' if i==0 else "")
        
        # If this is a reset point, draw the reset transition as a dashed line
        if i < len(reset_indices):
            reset_idx = reset_indices[i]
            if reset_idx+1 < len(states):
                # X phase reset
                ax3.plot([states[reset_idx, 0], states[reset_idx+1, 0]], 
                         [states[reset_idx, 2], states[reset_idx+1, 2]], 
                         'b--', alpha=0.7)
                # Y phase reset
                ax3.plot([states[reset_idx, 1], states[reset_idx+1, 1]], 
                         [states[reset_idx, 3], states[reset_idx+1, 3]], 
                         'r--', alpha=0.7)
    
    # Mark reset points on phase space
    for reset_idx in reset_indices:
        ax3.plot(states[reset_idx, 0], states[reset_idx, 2], 'k*', markersize=8)
        ax3.plot(states[reset_idx, 1], states[reset_idx, 3], 'k*', markersize=8)
    
    ax3.grid(True)
    ax3.set_xlabel('Position (m)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Phase Space')
    ax3.legend()
    
    # Synchronization measure plot (subplot 4)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Calculate synchronization measure for each point
    sync_measures = [model.synchronization_measure(state) for state in states]
    
    # Plot each continuous segment separately
    for i in range(len(segment_indices)-1):
        start_idx = segment_indices[i]
        end_idx = segment_indices[i+1]
        
        # Plot continuous segments
        ax4.plot(times[start_idx:end_idx+1], sync_measures[start_idx:end_idx+1], 
                'g-', label='Sync measure' if i==0 else "")
        
        # If this is a reset point, draw the reset transition as a dashed line
        if i < len(reset_indices):
            reset_idx = reset_indices[i]
            if reset_idx+1 < len(states):
                ax4.plot([times[reset_idx], times[reset_idx+1]], 
                         [sync_measures[reset_idx], sync_measures[reset_idx+1]], 
                         'g--', alpha=0.7)
    
    # Mark reset points on sync measure plot
    for reset_time in reset_times:
        ax4.axvline(x=reset_time, color='k', linestyle='--', alpha=0.5)
    
    ax4.grid(True)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Synchronization Measure L')
    ax4.set_title('Synchronization Measure vs Time')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Base parameters
    z0 = 1.0  # Height of the constraint plane
    r0 = 1.0  # Radius of the impact boundary
    
    # Define multiple initial conditions
    initial_conditions = [
        {
            "name": "Synchronized (θ=π/6)",
            "theta": np.pi/6,
            "vel_factor_x": 1.0,
            "vel_factor_y": 1.0,
            "color": 'b'
        },
        {
            "name": "Different angle (θ=π/4)",
            "theta": np.pi/4,
            "vel_factor_x": 1.0,
            "vel_factor_y": 1.0,
            "color": 'r'
        },
        {
            "name": "Faster x-velocity (θ=π/6)",
            "theta": np.pi/6,
            "vel_factor_x": 1.1,
            "vel_factor_y": 1.0,
            "color": 'g'
        },
        {
            "name": "Slower y-velocity (θ=π/6)",
            "theta": np.pi/6,
            "vel_factor_x": 1.0,
            "vel_factor_y": 0.9,
            "color": 'm'
        }
    ]
    
    # Simulation time
    t_max = 10.0
    
    # Store results for each initial condition
    results = []
    
    # Run simulations for each initial condition
    for ic in initial_conditions:
        # Calculate foothold position based on angle
        theta = ic["theta"]
        x0 = r0 * np.cos(theta)
        y0 = r0 * np.sin(theta)
        
        # Create model
        lip_model = LinearInvertedPendulum3D(x0, y0, z0)
        
        # Calculate synchronized velocity and apply factors
        v_sync = np.sqrt(lip_model.omega**2 * x0 * y0)
        v_x = v_sync * ic["vel_factor_x"]
        v_y = -v_sync * ic["vel_factor_y"]  # Negative for correct direction
        
        # Create initial state
        initial_state = np.array([-x0, y0, v_x, v_y])
        
        print(f"\nSimulating {ic['name']}:")
        print(f"  Initial state: {initial_state}")
        print(f"  Initial synchronization measure: {lip_model.synchronization_measure(initial_state):.6f}")
        
        # Simulate trajectory
        times, states, reset_times = simulate_trajectory(lip_model, initial_state, r0, t_max)
        
        print(f"  Simulation completed: {len(times)} time points, {len(reset_times)} resets")
        
        # Store results
        results.append({
            "name": ic["name"],
            "color": ic["color"],
            "times": times,
            "states": states,
            "reset_times": reset_times,
            "model": lip_model
        })
    
    # Plot all trajectories together
    plot_multiple_trajectories(results, r0)

def plot_multiple_trajectories(results, r0):
    """
    Plot multiple trajectories on the same set of axes
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing simulation results
    r0 : float
        Radius of the impact boundary
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Position x vs position y plot (subplot 1)
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Draw the circular impact boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(r0*np.cos(theta), r0*np.sin(theta), 'k--', alpha=0.5, label='Impact boundary')
    
    # Plot each trajectory
    for res in results:
        times = res["times"]
        states = res["states"]
        reset_times = res["reset_times"]
        color = res["color"]
        name = res["name"]
        
        # Find indices of reset points
        reset_indices = []
        for reset_time in reset_times:
            idx = np.where(times == reset_time)[0][0]
            reset_indices.append(idx)
        
        # Add start and end indices to create segments
        segment_indices = [0] + reset_indices + [len(times)-1]
        
        # Plot each continuous segment separately
        for i in range(len(segment_indices)-1):
            start_idx = segment_indices[i]
            end_idx = segment_indices[i+1]
            
            # For continuous segments, we need to be careful about the indices
            # If this is after a reset point, start from the point after the reset
            if i > 0:
                # This is after a reset, so start from the point after the reset
                plot_start_idx = start_idx + 1
            else:
                # This is the first segment, start from the beginning
                plot_start_idx = start_idx
                
            # Plot continuous segment
            ax1.plot(states[plot_start_idx:end_idx+1, 0], states[plot_start_idx:end_idx+1, 1], 
                     f'{color}-', linewidth=1.5, 
                     label=f'{name}' if i==0 else "")
            
            # If this is a reset point, draw the reset transition as a dashed line
            if i < len(reset_indices):
                reset_idx = reset_indices[i]
                if reset_idx+1 < len(states):
                    ax1.plot([states[reset_idx, 0], states[reset_idx+1, 0]], 
                             [states[reset_idx, 1], states[reset_idx+1, 1]], 
                             f'{color}--', linewidth=1.0, alpha=0.7)
        
        # Mark start and end points
        ax1.plot(states[0, 0], states[0, 1], f'{color}o', markersize=8)
        ax1.plot(states[-1, 0], states[-1, 1], f'{color}*', markersize=8)
    
    ax1.grid(True)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('X-Y Trajectories')
    ax1.set_aspect('equal')  # Equal aspect ratio
    ax1.legend()
    
    # Velocity plot (subplot 2)
    ax2 = fig.add_subplot(2, 2, 2)
    
    for res in results:
        times = res["times"]
        states = res["states"]
        reset_times = res["reset_times"]
        color = res["color"]
        name = res["name"]
        
        # Find indices of reset points
        reset_indices = []
        for reset_time in reset_times:
            idx = np.where(times == reset_time)[0][0]
            reset_indices.append(idx)
        
        # Add start and end indices to create segments
        segment_indices = [0] + reset_indices + [len(times)-1]
        
        # Plot each continuous segment separately
        for i in range(len(segment_indices)-1):
            start_idx = segment_indices[i]
            end_idx = segment_indices[i+1]
            
            # For continuous segments, we need to be careful about the indices
            # If this is after a reset point, start from the point after the reset
            if i > 0:
                # This is after a reset, so start from the point after the reset
                plot_start_idx = start_idx + 1
            else:
                # This is the first segment, start from the beginning
                plot_start_idx = start_idx
                
            # Plot continuous segment
            ax2.plot(times[plot_start_idx:end_idx+1], states[plot_start_idx:end_idx+1, 2], 
                    f'{color}-', label=f'{name} - ẋ' if i==0 else "")
            ax2.plot(times[plot_start_idx:end_idx+1], states[plot_start_idx:end_idx+1, 3], 
                    f'{color}--', alpha=0.5, label=f'{name} - ẏ' if i==0 else "")
    
    ax2.grid(True)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs Time')
    ax2.legend()
    
    # Phase space (subplot 3)
    ax3 = fig.add_subplot(2, 2, 3)
    
    for res in results:
        times = res["times"]
        states = res["states"]
        reset_times = res["reset_times"]
        color = res["color"]
        name = res["name"]
        
        # Find indices of reset points
        reset_indices = []
        for reset_time in reset_times:
            idx = np.where(times == reset_time)[0][0]
            reset_indices.append(idx)
        
        # Add start and end indices to create segments
        segment_indices = [0] + reset_indices + [len(times)-1]
        
        # Plot each continuous segment separately
        for i in range(len(segment_indices)-1):
            start_idx = segment_indices[i]
            end_idx = segment_indices[i+1]
            
            # For continuous segments, we need to be careful about the indices
            # If this is after a reset point, start from the point after the reset
            if i > 0:
                # This is after a reset, so start from the point after the reset
                plot_start_idx = start_idx + 1
            else:
                # This is the first segment, start from the beginning
                plot_start_idx = start_idx
                
            # Plot continuous segment
            ax3.plot(states[plot_start_idx:end_idx+1, 0], states[plot_start_idx:end_idx+1, 2], 
                    f'{color}-', label=f'{name}' if i==0 else "")
    
    ax3.grid(True)
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('X Velocity (m/s)')
    ax3.set_title('X Phase Space')
    ax3.legend()
    
    # Synchronization measure plot (subplot 4)
    ax4 = fig.add_subplot(2, 2, 4)
    
    for res in results:
        times = res["times"]
        states = res["states"]
        reset_times = res["reset_times"]
        color = res["color"]
        name = res["name"]
        model = res["model"]
        
        # Calculate synchronization measure for each point
        sync_measures = [model.synchronization_measure(state) for state in states]
        
        # Find indices of reset points
        reset_indices = []
        for reset_time in reset_times:
            idx = np.where(times == reset_time)[0][0]
            reset_indices.append(idx)
        
        # Add start and end indices to create segments
        segment_indices = [0] + reset_indices + [len(times)-1]
        
        # Plot each continuous segment separately
        for i in range(len(segment_indices)-1):
            start_idx = segment_indices[i]
            end_idx = segment_indices[i+1]
            
            # For continuous segments, we need to be careful about the indices
            # If this is after a reset point, start from the point after the reset
            if i > 0:
                # This is after a reset, so start from the point after the reset
                plot_start_idx = start_idx + 1
            else:
                # This is the first segment, start from the beginning
                plot_start_idx = start_idx
                
            # Plot continuous segment
            ax4.plot(times[plot_start_idx:end_idx+1], sync_measures[plot_start_idx:end_idx+1], 
                    f'{color}-', label=f'{name}' if i==0 else "")
    
    ax4.grid(True)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Synchronization Measure L')
    ax4.set_title('Synchronization Measure vs Time')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()