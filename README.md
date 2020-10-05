[//]: # (Image References)

[robotic_arm_environment]: robotic_arm_environment.png "Robotic arm environment"

# Udacity deep reinforcement learning course - project 2

### Introduction

For this project, I trained an agent to move a robotic arm with two joints and
make the gripper at the end stay in a moving target volume. The world can be
visualized in a Unity application. The agent is based on the PPO deep neural
network architecture.

![Robotic arm environment][robotic_arm_environment]

A reward of +0.1 is provided for each simulation step that the agents gripper is
inside the target volume.

Due to the continuous nature of the joint movement the state space is very
large. Basically just limited by the numeric precision. The agent can observe 33
values regarding position, rotation, velocity, and angular velocities of the two
arm rigid bodies. It can also observe the location of the target volume. Given
this information, the agent has to learn how to best select actions. As the next
action it can apply torque values to the two axis of each of the two joints.
Hence the size of the (vector) action space is 4.

The task is in principle continuing but the environment makes it episodic by
limiting the number of simulation steps. In order to solve the environment, the
agent must get an average score of +30 over 100 consecutive episodes.

### Unity environment download

1. Download the environment from one of the links below. You need only select
   the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. Extract it to a subdirectory.
3. Update the path to the executable in the Jupyter notebook.

### Python dependencies

Please make sure the following Python requirements are fulfilled in your environment:

- jupyter
- unityagents
- numpy
- matplotlib
- torch

This can be done with

    pip3 install -r requirements.txt

Or your own favorite way of Python dependency management.

### How to run

After downloading and extracting the Unity environment, execute the
[continuous_control.py](continuous_control.py) Python script in order to
train the agent and/or see the trained agent in action.

### Solution report

See [report](REPORT.md) for more information about my solution.
