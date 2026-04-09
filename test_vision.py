import gymnasium as gym
from ray.rllib.models.torch.visionnet import VisionNetwork
import torch

obs_space = gym.spaces.Box(0.0, 1.0, shape=(64, 64, 6))
act_space = gym.spaces.Discrete(14)
model_config = {
    "conv_filters": [
        [32, [8, 8], 4],
        [64, [4, 4], 2],
        [128, [8, 8], 1],
    ],
    "conv_activation": "relu",
    "fcnet_hiddens": [256, 128],
    "fcnet_activation": "relu",
}

net = VisionNetwork(obs_space, act_space, 14, model_config, "test")
x = torch.zeros(32, 64, 64, 6)
out, _ = net({"obs": x})
