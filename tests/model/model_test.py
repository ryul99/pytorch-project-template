import pytest
import torch
import torch.nn as nn
from model.model_arch import Net_arch
from model.model import Model
from utils.utils import load_hparam


class TestModel:
    @classmethod
    def setup_class(cls):
        cls.hp = load_hparam("config/default.yaml")
        cls.hp.model.device = "cpu"
        cls.net = Net_arch(cls.hp)
        cls.loss_f = nn.MSELoss()
        cls.model = Model(cls.hp, cls.net, cls.loss_f)

    def test_model(self):
        assert self.model.hp == self.hp
        assert self.model.net == self.net
        assert self.model.loss_f == self.loss_f

    def test_feed_data(self):
        input_ = torch.rand(64, 10)
        gt = torch.rand(64, 1)
        device_input_ = input_.to(self.hp.model.device)
        device_gt = gt.to(self.hp.model.device)
        self.model.feed_data(input=input_)
        assert (self.model.input == device_input_).all()
        self.model.feed_data(GT=gt)
        assert (self.model.GT == device_gt).all()
        self.model.feed_data()
        assert self.model.input is None
        assert self.model.GT is None
        self.model.feed_data(input=input_, GT=gt)
        assert (self.model.input == device_input_).all()
        assert (self.model.GT == device_gt).all()
