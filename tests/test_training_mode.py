import unittest
import torch
from torch import nn
import train


class TransferMode(unittest.TestCase):
    def test_frozen_statistics_still_allow_feature_learning(self):
        self.assertTrue(hasattr(train, 'set_training_mode'), 'training mode must support frozen BatchNorm statistics')
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1))
        train.set_training_mode(model, freeze_bn=True)
        before = model[1].running_mean.clone()
        model(torch.randn(2, 3, 8, 8)).sum().backward()
        self.assertTrue(torch.equal(before, model[1].running_mean))
        self.assertGreater(model[0].weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
