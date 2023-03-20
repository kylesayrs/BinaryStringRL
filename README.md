# Binary String Reinforcement Learning #
This is a personal project to learn more about reinforcement learning inspired by OpenAI's paper [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf). In this task an agent must perform bit flipping actions to transform a given binary string to a given target string with only sparse rewards.

This repo allows one to replicate the results found in the paper and adjust virtual goal parameters for better performance.

<p align="center">
    <img src="repo_assets/visualization.gif" alt="Visualization"/>
</p>

## Premise ##
The environment consists of a binary string of 1s and 0s with another binary goal string that the agent must transform the initial string into. For example, the optimal actions for a string of length 3 would look like this:
```
GOAL: [0 1 1]
STRING: [1 0 1] -> [0 0 1] -> [0 1 1]
```

With some domain knowledge, an implementer could create a shaped reward function that rewards an agent for taking actions that move the string closer to its goal (such as a norm function). However not all tasks have known or rewards capapble of being smoothly shaped for an agent. In [HER](https://arxiv.org/pdf/1707.01495.pdf) the authors consider the case in which the rewards are sparse: +1 reward if the strings match exactly, -1 reward otherwise.

A typical agent is not capapble of solving strings longer than 15 bits since it is statstically improbable that the agent would randomly encounter the goal enough to learn from positive rewards.

## Hindsight Experience Replay ##
In the paper the authors propose a method of adding additional goals for each replay step added to the replay buffer. For each action, in addition to the (probably negative) reward replay, they modify the replay's goal string to be the replay's next state, resulting in a replay with a postiive positive reward. These "virtual goals" allow the agent to learn from immediate positive rewards without relying on random sampling from the environment.


## Results
My results were able to replicate the authors' findings up to 30 bit strings. The authors did not include many details about the hyperparameters in their paper, so it is probable that longer strings are solvable with different parameters such as a larger replay buffer and more training episodes.
