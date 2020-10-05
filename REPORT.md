[//]: # (Image References)

[training_scores]: training_scores.png "Training_scores"

# Project 2: Continous control of robot arm - report

This report describes my solution and some findings along the way.

### Solution architecture and strategy

My solution uses a actor critic deep neural network architecture with a deep
deterministic policy gradient (DDPG) training approach. It is based on the
Udacity DDPG implementation used to solve the pendulum environment. After some
initial frustration to make the network learn anything useful. I found out that
the following changes where necessary to not get stuck at an average score
around 1:

- Gradient clipping in the critic network update step
- Batch normalization after the first layer in both networks as recommended by
  the community, blog post, and several other implementations

Furthermore, when I implemented the training with the environment with 20 Unity
robot arm agents I initially made the mistake to update the network, i.e. call
the learning method, for each and every of the 20 instances inside my loop where
I also stored the data samples into the replay buffer. Afterwards I found that

- Restructuring the DDPG agent code to be able to call the learning step and
  calling the learning step only once after filling the replay buffer with the
  20 new data points.

was the key to a successful training.

I then got a successful training result as seen in the plot below with the
environment being solved after about 100 Episodes. After some more episodes a
maximum average score of about 36 was reached.

![Training scores][training_scores]

The following parameters have been used for the submitted result:

- Replay buffer size: <img src="https://render.githubusercontent.com/render/math?math=1\times10^{-5}">
- Batch size: <img src="https://render.githubusercontent.com/render/math?math=128">
- Discount factor: <img src="https://render.githubusercontent.com/render/math?math=\gamma=0.99">
- Soft update coefficient: <img src="https://render.githubusercontent.com/render/math?math=1\times10^{-3}">
- Learning rate of actor: <img src="https://render.githubusercontent.com/render/math?math=1\times10^{-4}">
- Learning rate of critic: <img src="https://render.githubusercontent.com/render/math?math=1\times10^{-3}">
- Weight decay of Adam optimizer: <img src="https://render.githubusercontent.com/render/math?math=0">

### Findings

- It is very easy to introduce a seemingly very small error in the code that
  completely destroys the learning performance and makes it impossible to solve
  the environment. Debugging these kind of issues is much harder than in
  classical programming and algorithm design.
- I found it a bit weird that the description of the environment does not
  mention that the state contains information about the target volume position.
  This should probably be improved in a later iteration to give the students a
  better understanding of what problem the learned algorithm is actually
  solving. I believe solving the environment without the knowledge of the
  target position would require a much more sophisticated network architecture
  that is able to first search for the target and remember what it did several
  time steps in the past.

### Things to look at

In the future I might have a look at

- Newer state of the art algorithms as recommended in the course such as 
  - Trust Region Policy Optimization (TRPO)
  - Truncated Natural Policy Gradient (TNPG)
  - Proximal Policy Optimization (PPO)
  - Distributed Distributional Deterministic Policy Gradients (D4PG)
- Different network architectures that allow to track the target zone over time
  to improve the control of the arm even more.
