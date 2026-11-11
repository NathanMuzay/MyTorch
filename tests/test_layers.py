import unittest
import numpy as np
from src.neural_network.layers import Dense
from src.neural_network.activations import ReLU, Softmax
from src.neural_network.losses import CategoricalCrossEntropy
from src.neural_network.optimizers import SGD
from src.neural_network.model import Sequential


class TestLayersAndMath(unittest.TestCase):
    def test_dense_forward_backward(self):
        dense = Dense(in_features=4, out_features=2, weight_initializer="he")
        x = np.random.randn(3, 4)
        out = dense.forward(x)
        self.assertEqual(out.shape, (3, 2))

        # Backward
        grad_out = np.ones((3, 2))
        grad_in = dense.backward(grad_out)
        self.assertEqual(grad_in.shape, (3, 4))
        self.assertEqual(dense.grads["weights"].shape, (4, 2))
        self.assertEqual(dense.grads["bias"].shape, (1, 2))

    def test_relu(self):
        relu = ReLU()
        x = np.array([[-2.0, 0.0, 3.0]])
        out = relu.forward(x)
        np.testing.assert_array_equal(out, [[0.0, 0.0, 3.0]])
        grad = relu.backward(np.array([[1.0, 1.0, 1.0]]))
        np.testing.assert_array_equal(grad, [[0.0, 0.0, 1.0]])

    def test_softmax_properties(self):
        sm = Softmax()
        x = np.array([[1.0, 2.0, 3.0], [-10.0, 0.0, 10.0]])
        out = sm.forward(x)
        # Probabilities sum to 1
        sums = np.sum(out, axis=1)
        np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-6)

    def test_crossentropy_loss(self):
        cce = CategoricalCrossEntropy()
        preds = np.array([[0.1, 0.7, 0.2]])
        targets = np.array([[0.0, 1.0, 0.0]])
        loss = cce.forward(preds, targets)
        self.assertGreater(loss, 0.0)

    def test_model_save_and_load(self):
        import tempfile
        import os

        model = Sequential([
            Dense(5, 3, weight_initializer="he"),
            ReLU(),
            Dense(3, 2, weight_initializer="xavier"),
            Softmax(),
        ])
        x = np.random.randn(2, 5)
        pred1 = model.predict(x)

        with tempfile.NamedTemporaryFile(suffix=".nn", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path)
            loaded_model = Sequential.load(tmp_path)
            pred2 = loaded_model.predict(x)
            np.testing.assert_allclose(pred1, pred2, atol=1e-6)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_batchnorm1d(self):
        from src.neural_network.layers import BatchNorm1d

        bn = BatchNorm1d(num_features=4)
        np.random.seed(42)
        x = np.random.randn(20, 4) * 5.0 + 3.0  # Mean ~3, Std ~5

        # Forward training
        out = bn.forward(x, training=True)
        self.assertEqual(out.shape, (20, 4))
        # Batch normalized output should have mean close to 0 and variance close to 1
        np.testing.assert_allclose(np.mean(out, axis=0), np.zeros(4), atol=1e-5)
        np.testing.assert_allclose(np.var(out, axis=0), np.ones(4), atol=1e-3)

        # Backward pass
        grad_out = np.random.randn(20, 4)
        dx = bn.backward(grad_out)
        self.assertEqual(dx.shape, (20, 4))
        self.assertEqual(bn.grads["gamma"].shape, (1, 4))
        self.assertEqual(bn.grads["beta"].shape, (1, 4))

        # Forward eval (should use running stats)
        out_eval = bn.forward(x, training=False)
        self.assertEqual(out_eval.shape, (20, 4))

    def test_model_with_batchnorm_save_load(self):
        import tempfile
        import os
        from src.neural_network.layers import BatchNorm1d

        model = Sequential([
            Dense(8, 4),
            BatchNorm1d(4),
            ReLU(),
            Dense(4, 2),
            Softmax(),
        ])
        x = np.random.randn(5, 8)
        pred1 = model.predict(x)

        with tempfile.NamedTemporaryFile(suffix=".nn", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path)
            loaded = Sequential.load(tmp_path)
            self.assertEqual(len(loaded.layers), 5)
            self.assertIsInstance(loaded.layers[1], BatchNorm1d)
            pred2 = loaded.predict(x)
            np.testing.assert_allclose(pred1, pred2, atol=1e-6)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_lr_schedulers(self):
        from src.neural_network.optimizers import SGD
        from src.neural_network.schedulers import StepLR, ExponentialLR, CosineAnnealingLR

        # StepLR
        opt1 = SGD(learning_rate=0.1)
        sched1 = StepLR(opt1, step_size=2, gamma=0.5)  # initial step() called in __init__, epoch 0 -> 0.1
        self.assertAlmostEqual(opt1.learning_rate, 0.1)
        sched1.step()  # epoch 1 -> 0.1
        self.assertAlmostEqual(opt1.learning_rate, 0.1)
        sched1.step()  # epoch 2 -> 0.05
        self.assertAlmostEqual(opt1.learning_rate, 0.05)

        # ExponentialLR
        opt2 = SGD(learning_rate=0.1)
        sched2 = ExponentialLR(opt2, gamma=0.9)  # epoch 0 -> 0.1
        self.assertAlmostEqual(opt2.learning_rate, 0.1)
        sched2.step()  # epoch 1 -> 0.09
        self.assertAlmostEqual(opt2.learning_rate, 0.09)

        # CosineAnnealingLR
        opt3 = SGD(learning_rate=0.1)
        sched3 = CosineAnnealingLR(opt3, T_max=4, eta_min=0.01)  # epoch 0 -> 0.1
        self.assertAlmostEqual(opt3.learning_rate, 0.1)
        sched3.step()  # epoch 1
        sched3.step()  # epoch 2
        sched3.step()  # epoch 3
        sched3.step()  # epoch 4 -> reaches eta_min = 0.01
        self.assertAlmostEqual(opt3.learning_rate, 0.01, places=5)

    def test_focal_loss(self):
        from src.neural_network.losses import FocalLoss
        fl = FocalLoss(gamma=2.0)
        preds = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        loss = fl.forward(preds, targets)
        self.assertGreater(loss, 0.0)
        grad = fl.backward()
        self.assertEqual(grad.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()


