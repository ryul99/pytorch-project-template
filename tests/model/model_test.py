import pytest
import torch
import torch.nn as nn
from model.model_arch import Net_arch
from model.model import Model
from utils.utils import load_hparam


class TestModel:
    @classmethod
    def setup_class(cls):
        cls.input_ = torch.rand(64, 10)
        cls.gt = torch.rand(64, 1)

    def setup_method(self, method):
        self.hp = load_hparam("config/default.yaml")
        self.hp.model.device = "cpu"
        self.net = Net_arch(self.hp)
        self.loss_f = nn.MSELoss()
        self.model = Model(self.hp, self.net, self.loss_f)

    def test_model(self):
        assert self.model.hp == self.hp
        assert self.model.net == self.net
        assert self.model.loss_f == self.loss_f

    def test_feed_data(self):
        device_input_ = self.input_.to(self.hp.model.device)
        device_gt = self.gt.to(self.hp.model.device)
        self.model.feed_data(input=self.input_)
        assert (self.model.input == device_input_).all()
        self.model.feed_data(GT=self.gt)
        assert (self.model.GT == device_gt).all()
        self.model.feed_data()
        assert self.model.input is None
        assert self.model.GT is None
        self.model.feed_data(input=self.input_, GT=self.gt)
        assert (self.model.input == device_input_).all()
        assert (self.model.GT == device_gt).all()

    def test_run_network(self):
        self.model.feed_data(input=self.input_, GT=self.gt)
        output = self.model.run_network()
        assert output.shape == self.model.GT.shape

    def test_optimize_parameters(self):
        self.model.feed_data(input=self.input_, GT=self.gt)
        self.model.optimize_parameters()
        assert self.model.log.loss_v is not None

    def test_inference(self):
        self.model.feed_data(input=self.input_, GT=self.gt)
        output = self.model.inference()
        assert output.shape == self.model.GT.shape
