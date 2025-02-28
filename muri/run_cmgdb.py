import numpy as np

# Load the combined data
data = np.load('all_lip3d_states.npz')

# Access the arrays
states = data['states']           # All state vectors
traj_indices = data['traj_indices']  # Which trajectory each state belongs to
reset_flags = data['reset_flags']    # Which states are reset points (1=reset, 0=normal)

# # Example: Get all states from trajectory 5
# traj5_states = states[traj_indices == 5]

# # Example: Get all reset points
# reset_points = states[reset_flags == 1]

