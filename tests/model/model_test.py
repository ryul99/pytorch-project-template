import pytest
import torch
import torch.nn as nn
import os
import logging
from tests.test_case import ProjectTestCase
from model.model_arch import Net_arch
from model.model import Model


class TestModel(ProjectTestCase):
    @classmethod
    def setup_class(cls):
        cls.input_ = torch.rand(64, 10)
        cls.gt = torch.rand(64, 1)

    def setup_method(self, method):
        super(TestModel, self).setup_method()
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

    def test_save_load_network(self):
        local_net = Net_arch(self.hp)
        self.loss_f = nn.MSELoss()
        local_model = Model(self.hp, local_net, self.loss_f)

        self.model.save_network(self.logger)
        save_filename = "%s_%d.pt" % (self.hp.log.name, self.model.step)
        save_path = os.path.join(self.hp.log.chkpt_dir, save_filename)
        self.hp.load.network_chkpt_path = save_path

        assert os.path.exists(save_path) and os.path.isfile(save_path)
        assert os.path.exists(self.hp.log.log_file_path) and os.path.isfile(
            self.hp.log.log_file_path
        )

        local_model.load_network(logger=self.logger)
        parameters = zip(
            list(local_model.net.parameters()), list(self.model.net.parameters())
        )
        for load, origin in parameters:
            assert (load == origin).all()

    def test_save_load_state(self):
        local_net = Net_arch(self.hp)
        self.loss_f = nn.MSELoss()
        local_model = Model(self.hp, local_net, self.loss_f)

        self.model.save_training_state(self.logger)
        save_filename = "%s_%d.state" % (self.hp.log.name, self.model.step)
        save_path = os.path.join(self.hp.log.chkpt_dir, save_filename)
        self.hp.load.resume_state_path = save_path

        assert os.path.exists(save_path) and os.path.isfile(save_path)
        assert os.path.exists(self.hp.log.log_file_path) and os.path.isfile(
            self.hp.log.log_file_path
        )

        local_model.load_training_state(logger=self.logger)
        parameters = zip(
            list(local_model.net.parameters()), list(self.model.net.parameters())
        )
        for load, origin in parameters:
            assert (load == origin).all()
        assert local_model.epoch == self.model.epoch
        assert local_model.step == self.model.step
