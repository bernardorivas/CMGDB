import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class LinearInvertedPendulum3D:
    def __init__(self, z0, g=9.81):
        """
        Initialize the 3D Linear Inverted Pendulum model
        
        Parameters:
        -----------
        z0 : float
            Height of the constraint plane (constant height)
        g : float, optional
            Gravitational acceleration, default is 9.81 m/s^2
        """
        self.z0 = z0
        self.g = g
        self.omega = np.sqrt(g / z0)  # Natural frequency of the pendulum
    
    def dynamics(self, t, state):
        """
        Compute the dynamics of the 3D LIP model
        
        Parameters:
        -----------
        t : float
            Time variable (not used in autonomous systems but required by solver)
        state : array-like
            Current state [x, y, x_dot, y_dot]
            
        Returns:
        --------
        array-like
            State derivatives [x_dot, y_dot, x_ddot, y_ddot]
        """
        x, y, x_dot, y_dot = state
        
        # From equations in the paper:
        # ẍ = ω²x
        # ÿ = ω²y
        x_ddot = self.omega**2 * x
        y_ddot = self.omega**2 * y
        
        return [x_dot, y_dot, x_ddot, y_ddot]
    
    def reset_map(self, state):
        """
        Apply the discrete reset map when the pendulum transitions to a new support point
        
        Parameters:
        -----------
        state : array-like
            State before reset [x, y, x_dot, y_dot]
            
        Returns:
        --------
        array-like
            State after reset [x_new, y_new, x_dot_new, y_dot_new]
        """
        x, y, x_dot, y_dot = state
        
        # According to the discrete map in equation (1) from Razavi paper:
        # x⁺ = x⁻
        # y⁺ = -y⁻
        # ẋ⁺ = ẋ⁻
        # ẏ⁺ = -ẏ⁻
        
        return [x, -y, x_dot, -y_dot]
    
    def orbital_energy(self, state):
        """
        Calculate the orbital energies in x and y directions
        
        Parameters:
        -----------
        state : array-like
            Current state [x, y, x_dot, y_dot]
            
        Returns:
        --------
        tuple
            (Ex, Ey) orbital energies
        """
        x, y, x_dot, y_dot = state
        
        # From Definition 3 in Razavi paper:
        # Ex = ẋ² - ω²x²
        # Ey = ẏ² - ω²y²
        Ex = x_dot**2 - self.omega**2 * x**2
        Ey = y_dot**2 - self.omega**2 * y**2
        
        return (Ex, Ey)
    
    def synchronization_measure(self, state):
        """
        Calculate the synchronization measure
        
        Parameters:
        -----------
        state : array-like
            Current state [x, y, x_dot, y_dot]
            
        Returns:
        --------
        float
            Synchronization measure L = ẋẏ + ω²xy
        """
        x, y, x_dot, y_dot = state
        
        # From Definition 7 in Razavi paper:
        # L = ẋẏ + ω²xy
        L = x_dot * y_dot + self.omega**2 * x * y
        
        return L
    
    def simulate_step(self, initial_state, t_span, events=None):
        """
        Simulate one step of the 3D LIP model
        
        Parameters:
        -----------
        initial_state : array-like
            Initial state [x0, y0, x_dot0, y_dot0]
        t_span : tuple
            Time span for simulation (t_start, t_end)
        events : callable, optional
            Event function to terminate simulation
            
        Returns:
        --------
        OdeResult
            Result of the simulation
        """
        sol = solve_ivp(
            self.dynamics, 
            t_span, 
            initial_state, 
            method='RK45', 
            events=events,
            dense_output=True
        )
        
        return sol

# Define an impact event function (when the pendulum reaches a circular boundary)
def impact_event(r0):
    def event(t, state):
        x, y, _, _ = state
        return x**2 + y**2 - r0**2
    
    event.terminal = True  # Stop integration when event occurs
    event.direction = 1    # Only detect when crossing from inside to outside
    
    return event

# Simulate multiple steps with the (x0, y0)-invariant gait
def simulate_walking(lip_model, initial_state, num_steps, r0, max_time_per_step=5.0):
    """
    Simulate multiple steps of walking with the (x0, y0)-invariant gait
    
    Parameters:
    -----------
    lip_model : LinearInvertedPendulum3D
        The 3D LIP model
    initial_state : array-like
        Initial state [x0, y0, x_dot0, y_dot0]
    num_steps : int
        Number of steps to simulate
    r0 : float
        Radius for the impact condition (x² + y² = r0²)
    max_time_per_step : float, optional
        Maximum time to simulate for each step
        
    Returns:
    --------
    tuple
        (t_array, state_array) with time and state history
    """
    t_all = []
    states_all = []
    current_state = np.array(initial_state)
    t_offset = 0
    
    for step in range(num_steps):
        # Simulate until impact
        sol = lip_model.simulate_step(
            current_state, 
            (0, max_time_per_step), 
            events=impact_event(r0)
        )
        
        # Add time offset for continuity
        t_step = sol.t + t_offset
        t_all.extend(t_step)
        
        # Store states
        states_all.extend(sol.y.T)
        
        # Update time offset
        t_offset = t_step[-1]
        
        # Apply reset map and update current state for next step
        if step < num_steps - 1:
            current_state = lip_model.reset_map(sol.y.T[-1])
    
    return np.array(t_all), np.array(states_all)

# Create animation of the walking simulation
def animate_walking(t, states, r0, z0):
    """
    Create animation of the 3D LIP model walking
    
    Parameters:
    -----------
    t : array-like
        Time array
    states : array-like
        States array with shape (n, 4) where each row is [x, y, x_dot, y_dot]
    r0 : float
        Radius used for impact condition
    z0 : float
        Height of the constraint plane
        
    Returns:
    --------
    animation
        Matplotlib animation object
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    # Extract state components
    x = states[:, 0]
    y = states[:, 1]
    x_dot = states[:, 2]
    y_dot = states[:, 3]
    
    # Initialize plots
    pendulum_line, = ax1.plot([], [], [], 'b-', linewidth=2)
    pendulum_mass, = ax1.plot([], [], [], 'bo', markersize=10)
    trajectory, = ax1.plot([], [], [], 'r--', linewidth=1)
    
    state_x, = ax2.plot([], [], 'r-', label='x')
    state_y, = ax2.plot([], [], 'b-', label='y')
    ax2.legend()
    
    vel_x, = ax3.plot([], [], 'r-', label='x_dot')
    vel_y, = ax3.plot([], [], 'b-', label='y_dot')
    ax3.legend()
    
    phase_plot, = ax4.plot([], [], 'g-')
    phase_point, = ax4.plot([], [], 'go', markersize=8)
    
    # Circular boundary for impact
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = r0 * np.cos(theta)
    circle_y = r0 * np.sin(theta)
    ax4.plot(circle_x, circle_y, 'k--', alpha=0.5)
    
    # Set up axes
    ax1.set_xlim([-1.5*r0, 1.5*r0])
    ax1.set_ylim([-1.5*r0, 1.5*r0])
    ax1.set_zlim([0, 2*z0])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Linear Inverted Pendulum')
    
    ax2.set_xlim([0, t[-1]])
    ax2.set_ylim([min(min(x), min(y))*1.2, max(max(x), max(y))*1.2])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position')
    ax2.set_title('Position vs Time')
    
    ax3.set_xlim([0, t[-1]])
    ax3.set_ylim([min(min(x_dot), min(y_dot))*1.2, max(max(x_dot), max(y_dot))*1.2])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Velocity vs Time')
    
    ax4.set_xlim([-1.5*r0, 1.5*r0])
    ax4.set_ylim([-1.5*r0, 1.5*r0])
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('X-Y Plane Trajectory')
    ax4.grid(True)
    ax4.set_aspect('equal')
    
    def init():
        pendulum_line.set_data([], [])
        pendulum_line.set_3d_properties([])
        pendulum_mass.set_data([], [])
        pendulum_mass.set_3d_properties([])
        trajectory.set_data([], [])
        trajectory.set_3d_properties([])
        state_x.set_data([], [])
        state_y.set_data([], [])
        vel_x.set_data([], [])
        vel_y.set_data([], [])
        phase_plot.set_data([], [])
        phase_point.set_data([], [])
        return pendulum_line, pendulum_mass, trajectory, state_x, state_y, vel_x, vel_y, phase_plot, phase_point
    
    def animate(i):
        idx = min(i*5, len(t)-1)  # Speed up animation by skipping frames
        
        # Update pendulum
        pendulum_line.set_data([0, x[idx]], [0, y[idx]])
        pendulum_line.set_3d_properties([0, z0])
        pendulum_mass.set_data([x[idx]], [y[idx]])
        pendulum_mass.set_3d_properties([z0])
        
        # Update trajectory
        trajectory.set_data(x[:idx+1], y[:idx+1])
        trajectory.set_3d_properties([z0] * (idx+1))
        
        # Update state plots
        state_x.set_data(t[:idx+1], x[:idx+1])
        state_y.set_data(t[:idx+1], y[:idx+1])
        
        # Update velocity plots
        vel_x.set_data(t[:idx+1], x_dot[:idx+1])
        vel_y.set_data(t[:idx+1], y_dot[:idx+1])
        
        # Update phase plot
        phase_plot.set_data(x[:idx+1], y[:idx+1])
        phase_point.set_data([x[idx]], [y[idx]])
        
        return pendulum_line, pendulum_mass, trajectory, state_x, state_y, vel_x, vel_y, phase_plot, phase_point
    
    ani = FuncAnimation(fig, animate, frames=len(t)//5, init_func=init, blit=True, interval=30)
    plt.tight_layout()
    
    return ani

# Main simulation function
def main():
    # Parameters
    z0 = 1.0  # Height of the constraint plane
    x0 = 0.3  # Position in x direction
    y0 = 0.15  # Position in y direction
    r0 = np.sqrt(x0**2 + y0**2)  # Radius for impact
    
    # Create model
    lip_model = LinearInvertedPendulum3D(z0)
    
    # Initial conditions for (x0, y0)-invariant gait
    # Starting with x = -x0, y = y0 as specified in Definition 1
    # Initialize velocities to achieve synchronization (L0 = 0)
    v0 = 0.7  # Initial speed
    gamma0 = -lip_model.omega**2 * (-x0) * y0 / v0  # For synchronization
    
    # Convert to cartesian velocities
    theta0 = np.arctan2(-x0, y0)
    x_dot0 = v0 * np.sin(theta0)
    y_dot0 = v0 * np.cos(theta0)
    
    # Ensure synchronization measure is close to zero
    initial_state = [-x0, y0, x_dot0, y_dot0]
    L0 = lip_model.synchronization_measure(initial_state)
    print(f"Initial synchronization measure: {L0}")
    
    # Simulate walking for multiple steps
    num_steps = 10
    t_array, states_array = simulate_walking(lip_model, initial_state, num_steps, r0)
    
    # Create and show animation
    ani = animate_walking(t_array, states_array, r0, z0)
    
    # Calculate orbital energies for each step
    step_indices = np.where(np.diff(np.sign(states_array[:, 1])) != 0)[0]
    print("\nOrbital energies at step transitions:")
    for i, idx in enumerate(step_indices):
        if i < len(step_indices) - 1:
            state = states_array[idx]
            Ex, Ey = lip_model.orbital_energy(state)
            L = lip_model.synchronization_measure(state)
            print(f"Step {i+1}: Ex = {Ex:.4f}, Ey = {Ey:.4f}, L = {L:.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()