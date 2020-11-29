import os
import tempfile

import torch
from hydra.experimental import compose, initialize

from model.model_arch import Net_arch

TEST_DIR = tempfile.mkdtemp(prefix="project_tests")


def test_net_arch():
    os.makedirs(TEST_DIR, exist_ok=True)
    with initialize(config_path="../../config"):
        cfg = compose(config_name="default", overrides=[f"working_dir={TEST_DIR}"])

    net = Net_arch(cfg)

    # TODO: This is example code. You should change this part as you need. You can code this part as forward
    x = torch.rand(8, 1, 28, 28)
    x = net.conv1(x)  # x: (B,4,14,14)
    assert x.shape == (8, 4, 14, 14)
    x = net.conv2(x)  # x: (B,4,7,7)
    assert x.shape == (8, 4, 7, 7)
    x = torch.flatten(x, 1)  # x: (B,4*7*7)
    assert x.shape == (8, 4 * 7 * 7)
    x = net.fc(x)  # x: (B,10)
    assert x.shape == (8, 10)
