import unittest
import torch
from model import BasicBlock, ResNet18

class NetworkBehavior(unittest.TestCase):
    def test_output_and_residual_gradients(self):
        model = ResNet18(num_classes=12).eval()
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 12))
        output.sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertGreater(model.layer2[0].downsample[0].weight.grad.abs().sum().item(), 0)

if __name__ == '__main__':
    unittest.main()
