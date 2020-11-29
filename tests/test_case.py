# ref: https://github.com/allenai/allennlp/blob/9c51d6c89875b3a3a50cac165d6f3188d9941c5b/allennlp/common/testing/test_case.py

import os
import pathlib
import shutil
import tempfile

from hydra.experimental import compose, initialize
from omegaconf import OmegaConf, open_dict

from utils.utils import get_logger

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

        # load job_logging_cfg
        project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hydra_conf = OmegaConf.load(
            os.path.join(project_root_path, "config/default.yaml")
        )
        job_logging_name = None
        for job_logging in hydra_conf.defaults:
            job_logging_name = job_logging.get("hydra/job_logging")
            if job_logging_name is not None:
                break
        job_logging_cfg_path = os.path.join(
            project_root_path,
            "config/hydra/job_logging",
            str(job_logging_name) + ".yaml",
        )
        if os.path.exists(job_logging_cfg_path):
            job_logging_cfg = OmegaConf.load(job_logging_cfg_path)
        else:
            job_logging_cfg = dict()
        with open_dict(self.cfg):
            self.cfg.job_logging_cfg = job_logging_cfg
        self.cfg.job_logging_cfg.handlers.file.filename = str(
            (self.working_dir / "trainer.log").resolve()
        )
        # set logger
        self.logger = get_logger(self.cfg, os.path.basename(__file__))

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
