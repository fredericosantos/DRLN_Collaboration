[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/fredericosantos/DRLN_ContinuousControl/master/Graph.png "Scores"

# Report on Project 1: Navigation

## Learning Algorithm

### Learning Algorithm

I used the MADDPG algorithm for this project. I used the recently released Rectified Adam as my optimizer.

### Hyperparameters

The hyperparameters values I selected were:

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.001        # learning rate of the actor
LR_CRITIC = 0.001       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # learning timestep interval
UPDATE_NUM = 2          # number of learning passes
RANDOM_SEED = 0         # Random Seed
NOISE_START = 1.0       # Noise value at start of training
T_NOISE_STOP = 1000     # t step where noise stops
NOISE_DECAY = 0.9999    # Noise decay every t step


### Model Architecture

Both Actor and Critic models are composed of 3 fully connected layers, with a 1D batch normalization layer from the first to second layer.

## Plot of Rewards

![Scores][image1]


## Ideas for Future Work

In the future I would like to be able to implement more advanced algorithms such as the MAD4PG that's more recent.

I would also like to work to implement Prioritized Experienced Replay so the Agent has a better replay pool.

I would also like to be able to implement the newly released DeepMind algorithm TVT!





