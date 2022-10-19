# -----------------------------------------------------------------------------
#           Simple maze problem
#       Goal is to find shortest path to from current state to room 5
#------------------------------------------------------------------------------
# MODIFIED LINES FOR SOLUTION: 11, 25, 44 TO 48, 90
import numpy as np

# Initialization

# Reward Matrix R
R = np.matrix([[-1, 0, -1, -1, 0, 100],
               [0, -1, -1, 0, -1, -1],
               [-1, -1, -1, 0, -1, -1],
               [-1, 0, 0, -1, 0, -1],
               [0, -1, -1, 0, -1, 100],
               [0, -1, -1, -1, 0, 100]])

# Q-value matrix
Q = np.matrix(np.zeros([6,6]))

# Gammaa
g = 0.8

# Eta
eta = 0.2

# Initial state (usually chosen at random)
initial_state = 1

# -----------------------------------------------------------------------------
#           Creating Q-learning algorithm functions
# -----------------------------------------------------------------------------
# Create function that provides avaialble actions from current state
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]   # -1 indicates no available edge to traverse
    return av_act

# Get avaialable actions from the current state
available_act = available_actions(initial_state)

# Create function that chooses next action in random within the range of all available states
def sample_next_action(available_actions_range):
    rn = np.random.random()
    if rn < eta:
        next_action = np.random.randint(6)
    else :
        next_action = int(np.random.choice(available_act, 1))
    return next_action

# Sample next action to be performed
action = sample_next_action(available_act)

# Create function to update Q-value matrix according to the path selected and Q-learning algorithm
def update(current_state, action, g):
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]
    
    if max_index.shape[0]>1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    
    # Q learning formula
    Q[current_state, action] = R[current_state, action] + g * max_value
    
# Update Q-value matrix
update(initial_state, action, g)

# -----------------------------------------------------------------------------
#           Training Markov (Number of episodes is 10000)
# -----------------------------------------------------------------------------

for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state, action, g)

# Normalize the "trained" Q-value matrix.
# Sometimes value in Q-value matrix go huge, which takes huge computation. so,
# to cut down to small we use this
print("Trained Q matrix: ")
print(Q/np.max(Q) * 100)

# -----------------------------------------------------------------------------
#          Getting the output from trained Q-value matrix
# -----------------------------------------------------------------------------
# current_state = 1            # figure out how to go from room 1 to room 5
current_state = 3            # figure out how to go from room 2 to room 5

steps = [current_state]

while current_state != 5:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
        
    steps.append(next_step_index)
    current_state = next_step_index
    
# print the output of path that leads to room 5
print("Selected path: ")
print(steps)




