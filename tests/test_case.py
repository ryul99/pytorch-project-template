# ref: https://github.com/allenai/allennlp/blob/9c51d6c89875b3a3a50cac165d6f3188d9941c5b/allennlp/common/testing/test_case.py

import os
import pathlib
import shutil
import tempfile
from utils.utils import get_logger
from hydra.experimental import initialize, compose

TEST_DIR = tempfile.mkdtemp(prefix="project_tests")


class ProjectTestCase:
    def setup_method(self):
        # set log/checkpoint dir
        self.TEST_DIR = pathlib.Path(TEST_DIR)
        self.working_dir = self.TEST_DIR
        chkpt_dir = (self.TEST_DIR / "chkpt").resolve()
        os.makedirs(self.TEST_DIR, exist_ok=True)
        os.makedirs(chkpt_dir, exist_ok=True)

        # set cfg
        with initialize(config_path="../config"):
            self.cfg = compose(
                config_name="default", overrides=[f"working_dir={self.working_dir}"]
            )
        self.cfg.device = "cpu"
        self.cfg.log.chkpt_dir = str(chkpt_dir)
        self.cfg.log.use_wandb = False
        self.cfg.log.use_tensorboard = False

        # set logger
        self.logger = get_logger(
            self.cfg,
            os.path.basename(__file__),
            str((self.working_dir / "trainer.log").resolve()),
        )

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
