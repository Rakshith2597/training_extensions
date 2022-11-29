import unittest
import os
from src.utils.train_utils import train_model
from src.utils.model import Encoder, Decoder
from src.utils.downloader import download_checkpoint, download_data
from src.utils.get_config import get_config


def create_train_test_for_phase1():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train', phase=1)
            cls.config = config
            if not os.path.exists(config["image_path"]):
                download_data(phase=1)

        def test_trainer(self):
            self.model = Encoder(n_downconv=self.config["n_digestunits"])
            if not os.path.exists(self.config["checkpoint"]):
                download_checkpoint(phase=1)
            self.device = self.config["device"]
            avg_loss1, avg_ssim1, avg_psnr1 = train_model(self.config)
            avg_loss2, avg_ssim2, avg_psnr2 = train_model(self.config)
            self.assertLessEqual(avg_loss2, avg_loss1)

        def test_config(self):
            self.config = get_config(action='train', phase=1)
            self.assertGreaterEqual(self.config["lr"], 1e-8)
            self.assertGreaterEqual(self.config['alpha'], 0)
            self.assertGreaterEqual(self.config['phi'], -1)
            self.assertLessEqual(self.config['alpha'], 2)
            self.assertLessEqual(self.config['phi'], 1)

    return TrainerTest

def create_train_test_for_phase2():
    class TrainerTestEff(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train', phase=2)
            cls.config = config
            if not os.path.exists(config["image_path"]):
                download_data(phase=2)

        def test_trainer(self):
            self.model = Decoder(n_upconv=self.config['n_digestunits'])
            if not os.path.exists(self.config["checkpoint"]):
                download_checkpoint(phase=2)
            self.device = self.config["device"]
            avg_loss1, avg_ssim1, avg_psnr1 = train_model(self.config)
            avg_loss2, avg_ssim2, avg_psnr2 = train_model(self.config)
            self.assertLessEqual(avg_loss2, avg_loss1)

        def test_config(self):
            self.config = get_config(action='train', phase=2)
            self.assertGreaterEqual(self.config["lr"], 1e-8)
            self.assertGreaterEqual(self.config['alpha'], 0)
            self.assertGreaterEqual(self.config['phi'], -1)
            self.assertLessEqual(self.config['alpha'], 2)
            self.assertLessEqual(self.config['phi'], 1)

    return TrainerTestEff


class TestTrainer(create_train_test_for_phase1()):
    'Test case for phase1'


class TestTrainerEff(create_train_test_for_phase2()):
    'Test case for phase2'


if __name__ == '__main__':

    unittest.main()