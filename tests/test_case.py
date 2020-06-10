# ref: https://github.com/allenai/allennlp/blob/9c51d6c89875b3a3a50cac165d6f3188d9941c5b/allennlp/common/testing/test_case.py

import os
import pathlib
import shutil
import tempfile
from utils.utils import load_hparam
from utils.logger import make_logger

TEST_DIR = tempfile.mkdtemp(prefix="project_tests")


class ProjectTestCase:
    """
    A custom subclass of `unittest.TestCase` that disables some of the more verbose AllenNLP
    logging and that creates and destroys a temp directory as a test fixture.
    """

    def setup_method(self):
        # set log/checkpoint dir
        self.TEST_DIR = pathlib.Path(TEST_DIR)
        self.log_dir = (self.TEST_DIR / "logs").resolve()
        self.chkpt_dir = (self.TEST_DIR / "chkpt").resolve()
        os.makedirs(self.TEST_DIR, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.chkpt_dir, exist_ok=True)

        # set hp
        self.hp = load_hparam("config/default.yaml")
        self.hp.model.device = "cpu"
        self.hp.log.log_dir = self.log_dir
        self.hp.log.chkpt_dir = self.chkpt_dir
        self.hp.log.use_wandb = False
        self.hp.log.use_tensorboard = False

        # set logger
        self.logger = make_logger(self.hp)

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
