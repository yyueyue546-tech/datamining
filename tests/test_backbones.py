import importlib.util
from pathlib import Path
import unittest

import torch
from torch_geometric.nn import SAGEConv

from models import GCN, GraphSAGE


class BackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(6, 4, dtype=torch.float64)
        # Multiple neighbors, duplicate edge, self-loop and an isolated node.
        self.edges = torch.tensor([[0, 1, 0, 2, 3, 4], [2, 2, 2, 2, 4, 3]])

    def test_local_gcn_parity(self):
        path = Path(__file__).resolve().parents[1] / (
            "datamining/TT_GREB_Reproduce/tt_greb_reproduce/models.py"
        )
        spec = importlib.util.spec_from_file_location("handoff_models", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        reference = original.GCN(4, 8, 3).double()
        model = GCN(4, 8, 3).double()
        for new, old in zip(model.convs, (reference.conv1, reference.conv2)):
            new.load_state_dict(old.state_dict())
        for training in (False, True):
            model.train(training)
            reference.train(training)
            for weights in (None, torch.linspace(0.2, 1.2, 6, dtype=self.x.dtype)):
                torch.manual_seed(100)
                expected = reference(self.x, self.edges, weights, return_embedding=True)
                torch.manual_seed(100)
                actual = model.forward_with_embeddings(self.x, self.edges, weights)
                for result, target in zip(actual, expected):
                    torch.testing.assert_close(result, target, rtol=0, atol=0)

    def test_sage_native_and_unit_weight_gradients(self):
        model = GraphSAGE(4, 8, 3, num_layers=1).double()
        native = SAGEConv(4, 3).double()
        native.load_state_dict(model.convs[0].state_dict(), strict=True)
        x = self.x.requires_grad_()
        expected = native(x, self.edges)
        torch.testing.assert_close(model(x, self.edges), expected, rtol=0, atol=0)
        reference_grad = torch.autograd.grad(expected.square().sum(), (x, *native.parameters()))
        weights = torch.ones(self.edges.size(1), dtype=x.dtype, requires_grad=True)
        actual = model(x, self.edges, weights)
        torch.testing.assert_close(actual, expected)
        gradients = torch.autograd.grad(actual.square().sum(), (x, *model.parameters(), weights))
        for result, target in zip(gradients[:-1], reference_grad):
            torch.testing.assert_close(result, target)
        self.assertTrue(torch.isfinite(gradients[-1]).all())
        self.assertGreater(gradients[-1].abs().sum().item(), 0)
        zero_output = model(x, self.edges, torch.zeros_like(weights))
        conv = model.convs[0]
        torch.testing.assert_close(zero_output, conv.lin_l(torch.zeros_like(x)) + conv.lin_r(x))

    def test_training_and_frozen_input_gradients(self):
        for backbone in (GCN, GraphSAGE):
            with self.subTest(backbone=backbone.__name__):
                model = backbone(4, 8, 3, num_layers=3, batch_norm=True).double()
                logits, embeddings = model.forward_with_embeddings(self.x, self.edges)
                self.assertEqual(logits.shape, (6, 3))
                self.assertEqual(embeddings.shape, (6, 8))
                logits.square().sum().backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all()
                                    for p in model.parameters()))
                model.reset_parameters()
                self.assertEqual(model.norms[0].num_batches_tracked.item(), 0)
                model.zero_grad(set_to_none=True)
                model.eval().requires_grad_(False)
                x = self.x.clone().requires_grad_()
                weights = torch.ones(6, dtype=x.dtype, requires_grad=True)
                logits, embeddings = model.forward_with_embeddings(x, self.edges, weights)
                torch.testing.assert_close(embeddings, model.get_embeddings(x, self.edges, weights))
                torch.testing.assert_close(logits, model(x, self.edges, weights))
                changed = weights.detach().clone()
                changed[0] = 0.2
                self.assertFalse(torch.allclose(logits, model(x, self.edges, changed)))
                logits.square().sum().backward()
                for gradient in (x.grad, weights.grad):
                    self.assertTrue(gradient.isfinite().all())
                    self.assertGreater(gradient.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_dataset_shaped_inputs(self):
        # Representative input sizes, independent of dataset loading or provenance.
        for name, inputs, classes, layers in (
            ("Cora", 1443, 7, 2),
            ("Amazon-Photo", 755, 8, 2),
            ("Twitch-E", 3170, 2, 2),
            ("OGB-Arxiv", 128, 40, 5),
        ):
            for backbone in (GCN, GraphSAGE):
                with self.subTest(dataset=name, backbone=backbone.__name__):
                    model = backbone(inputs, 32, classes, layers, batch_norm=True).eval()
                    logits, embeddings = model.forward_with_embeddings(
                        torch.randn(6, inputs), self.edges
                    )
                    self.assertEqual(logits.shape, (6, classes))
                    self.assertEqual(embeddings.shape, (6, 32))


if __name__ == "__main__":
    unittest.main()
