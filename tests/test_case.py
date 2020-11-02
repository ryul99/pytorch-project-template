# ref: https://github.com/allenai/allennlp/blob/9c51d6c89875b3a3a50cac165d6f3188d9941c5b/allennlp/common/testing/test_case.py

import os
import pathlib
import shutil
import tempfile
from utils.logger import make_logger
from hydra.experimental import initialize, compose

TEST_DIR = tempfile.mkdtemp(prefix="project_tests")


class ProjectTestCase:
    def setup_method(self):
        # set log/checkpoint dir
        self.TEST_DIR = pathlib.Path(TEST_DIR)
        self.log_dir = (self.TEST_DIR / "logs").resolve()
        self.chkpt_dir = (self.TEST_DIR / "chkpt").resolve()
        os.makedirs(self.TEST_DIR, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.chkpt_dir, exist_ok=True)

        # set cfg
        with initialize(config_path="../config"):
            self.cfg = compose(config_name="default")
        self.cfg.model.device = "cpu"
        self.cfg.log.log_dir = str(self.log_dir)
        self.cfg.log.chkpt_dir = str(self.chkpt_dir)
        self.cfg.log.use_wandb = False
        self.cfg.log.use_tensorboard = False

        # set logger
        self.logger = make_logger(self.cfg)

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
