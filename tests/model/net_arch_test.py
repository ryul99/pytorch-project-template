import torch
from model.model_arch import Net_arch
from utils.utils import load_hparam


def test_net_arch():
    hp = load_hparam("config/default.yaml")
    net = Net_arch(hp)

    # TODO: This is example code. You should change this part as you need. You can code this part as forward
    x = torch.rand(64, 10)
    x = net.fc1(x)
    assert x.shape == (64, 10)
    x = net.fc2(x)
    assert x.shape == (64, 1)
