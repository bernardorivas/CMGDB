import numpy as np
from scipy.integrate import solve_ivp

class LinearInvertedPendulum3D:
    def __init__(self, x0, y0, z0, g=9.81):
        """
        Initialize the 3D Linear Inverted Pendulum model
        
        Parameters:
        -----------
        x0, y0 : float
            Parameters defining the foot hold
        z0 : float
            Height of the constraint plane
        g : float, optional
            Gravitational acceleration, default is 9.81 m/s^2
        """
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.g = g
        self.omega = np.sqrt(g / z0)  # Natural frequency of the pendulum
    
    def dynamics(self, t, state):
        """
        Compute the dynamics of the 3D LIP model
        
        Parameters:
        -----------
        t : float
            Time variable
        state : array-like
            Current state [x, y, x_dot, y_dot]
            
        Returns:
        --------
        array-like
            State derivatives [x_dot, y_dot, x_ddot, y_ddot]
        """
        x, y, x_dot, y_dot = state
        
        # From equations in the paper: x'' = ω²x, y'' = ω²y
        x_ddot = self.omega**2 * x
        y_ddot = self.omega**2 * y
        
        return [x_dot, y_dot, x_ddot, y_ddot]
    
    def reset_map(self, state):
        """
        Apply the discrete reset map according to the (x0, y0)-invariant gait definition
        
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

        # According to Definition 1, (x0,y0)-invariant step if at the moment of double support (x_FH, y_FH) = (-x0, -y0)
        x_FH = -self.x0
        y_FH = -self.y0
        
        # Equation (1) in the paper defines the reset map
        # return [x_FH, -y_FH, x_dot, -y_dot]
        # return [x_FH, -y_FH, -x_dot, y_dot]
        return [x, -y, -x_dot, y_dot]
        # return [x, -y, x_dot, -y_dot]
    
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
        
        # From Definition 3 in Razavi paper: Ex = ẋ² - ω²x², Ey = ẏ² - ω²y²
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
        
        # From Definition 7 in Razavi paper: L = ẋẏ + ω²xy assuming x = -x0
        L = x_dot * y_dot + self.omega**2 * (-x) * y
        
        return L


def impact_event(r0, min_time=0.05):
    """
    Define the impact event function for the 3D LIP model
    
    Parameters:
    -----------
    r0 : float
        Radius of the impact boundary
    min_time : float, optional
        Minimum time before impact can be detected, default is 0.05s
        
    Returns:
    --------
    function
        Event function for solve_ivp
    """
    def event(t, state):
        x, y, _, _ = state
        # The switching manifold is defined as x² + y² = r0²
        dist_from_boundary = x**2 + y**2 - r0**2
        
        # Only consider impacts when crossing from inside to outside
        # and after minimum time has passed
        if t < min_time:
            return 1.0
            
        return dist_from_boundary
    
    # Terminal event (stop integration when triggered)
    event.terminal = True
    
    # Direction: +1 means detect when function goes from negative to positive
    # This means we detect when the COM moves from inside to outside the circle
    event.direction = 1
    
    return event
