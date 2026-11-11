"""MLP pédagogique : tanh, softmax, entropie croisée, SGD, sans dépendance.

Convention : W[i][j] relie l'entrée i à la sortie j.
Ce script apprend XOR ; il ne fournit pas un modèle d'échecs.
"""

import json
import math
import random
import tempfile
from pathlib import Path


def softmax(logits):
    maximum = max(logits)
    values = [math.exp(value - maximum) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def cross_entropy(logits, target):
    """Loss stable sur les logits, sans calculer log d'une probabilité nulle."""
    maximum = max(logits)
    return math.log(sum(math.exp(z - maximum) for z in logits)) - (
        logits[target] - maximum
    )


class MLP:
    def __init__(self, sizes, seed=7):
        self.sizes = list(sizes)
        rng = random.Random(seed)
        self.weights = []
        self.biases = []
        for inputs, outputs in zip(sizes, sizes[1:]):
            sigma = math.sqrt(2 / (inputs + outputs))  # Xavier normal
            self.weights.append([
                [rng.gauss(0, sigma) for _ in range(outputs)]
                for _ in range(inputs)
            ])
            self.biases.append([0.0] * outputs)

    def forward(self, x):
        """Renvoie logits et cache des activations pour un seul exemple."""
        activations = [list(x)]
        for layer, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            previous = activations[-1]
            z = [
                biases[j] + sum(previous[i] * weights[i][j]
                                for i in range(len(previous)))
                for j in range(len(biases))
            ]
            if layer < len(self.weights) - 1:
                activations.append([math.tanh(value) for value in z])
            else:
                activations.append(z)  # Scores linéaires à la sortie
        return activations[-1], activations

    def loss(self, examples):
        return sum(cross_entropy(self.forward(x)[0], y)
                   for x, y in examples) / len(examples)

    def loss_and_grad(self, examples):
        """Gradients de la perte MOYENNE ; aucune mise à jour ici."""
        dw = [[[0.0 for _ in row] for row in weights]
              for weights in self.weights]
        db = [[0.0 for _ in biases] for biases in self.biases]
        total_loss = 0.0
        batch_size = len(examples)
        for x, target in examples:
            logits, activations = self.forward(x)
            total_loss += cross_entropy(logits, target)
            probabilities = softmax(logits)
            gradient = [
                (p - (1.0 if j == target else 0.0)) / batch_size
                for j, p in enumerate(probabilities)
            ]
            # gradient porte sur les logits : softmax est déjà différenciée.
            for layer in reversed(range(len(self.weights))):
                previous = activations[layer]
                weights = self.weights[layer]
                for i in range(len(previous)):
                    for j in range(len(gradient)):
                        dw[layer][i][j] += previous[i] * gradient[j]
                for j in range(len(gradient)):
                    db[layer][j] += gradient[j]
                if layer > 0:
                    # W n'a pas encore changé : on remonte le même forward.
                    upstream = [
                        sum(weights[i][j] * gradient[j]
                            for j in range(len(gradient)))
                        for i in range(len(previous))
                    ]
                    gradient = [g * (1 - a * a)
                                for g, a in zip(upstream, previous)]
        return total_loss / batch_size, dw, db

    def step(self, dw, db, learning_rate):
        for layer, weights in enumerate(self.weights):
            for i, row in enumerate(weights):
                for j in range(len(row)):
                    row[j] -= learning_rate * dw[layer][i][j]
            for j in range(len(self.biases[layer])):
                self.biases[layer][j] -= learning_rate * db[layer][j]

    def probabilities(self, x):
        return softmax(self.forward(x)[0])

    def save(self, path):
        payload = {
            "version": 1,
            "sizes": self.sizes,
            "hidden_activation": "tanh",
            "labels": [0, 1],
            "weights": self.weights,
            "biases": self.biases,
        }
        Path(path).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (payload["version"] != 1 or payload["hidden_activation"] != "tanh"
                or payload["labels"] != [0, 1]):
            raise ValueError("Format incompatible")
        sizes = payload["sizes"]
        if (len(sizes) < 2 or sizes[-1] != 2
                or any(type(size) is not int or size <= 0 for size in sizes)):
            raise ValueError("Architecture invalide")
        model = cls(sizes)
        weights, biases = payload["weights"], payload["biases"]
        if len(weights) != len(sizes) - 1 or len(biases) != len(weights):
            raise ValueError("Nombre de couches incohérent")
        for layer, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
            if (len(weights[layer]) != inputs or len(biases[layer]) != outputs
                    or any(len(row) != outputs for row in weights[layer])):
                raise ValueError("Dimensions incohérentes")
            values = [value for row in weights[layer] for value in row]
            values += biases[layer]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Paramètre non fini")
        model.weights, model.biases = weights, biases
        return model


def gradient_check(model, examples, epsilon=1e-5):
    """Différences centrées pour TOUS les poids et biais du petit modèle."""
    _, dw, db = model.loss_and_grad(examples)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    parameters = []
    for layer, weights in enumerate(model.weights):
        for i, row in enumerate(weights):
            parameters.extend((row, j, dw[layer][i][j]) for j in range(len(row)))
        parameters.extend((model.biases[layer], j, db[layer][j])
                          for j in range(len(model.biases[layer])))
    for container, index, analytic in parameters:
        original = container[index]
        try:
            container[index] = original + epsilon
            plus = model.loss(examples)
            container[index] = original - epsilon
            minus = model.loss(examples)
        finally:
            container[index] = original
        numeric = (plus - minus) / (2 * epsilon)
        absolute = abs(analytic - numeric)
        relative = absolute / max(1e-8, abs(analytic) + abs(numeric))
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        if absolute > 1e-7 + 1e-4 * max(abs(analytic), abs(numeric)):
            raise AssertionError(f"Gradient incorrect : {analytic} / {numeric}")
    return len(parameters), maximum_absolute, maximum_relative


def main():
    examples = [([0.0, 0.0], 0), ([0.0, 1.0], 1),
                ([1.0, 0.0], 1), ([1.0, 1.0], 0)]
    assert all(abs(p - 1 / 3) < 1e-12 for p in softmax([1000.0] * 3))
    assert abs(cross_entropy([1000.0] * 3, 0) - math.log(3)) < 1e-12
    model = MLP([2, 8, 2])
    count, absolute, relative = gradient_check(model, examples)
    print(f"Gradients vérifiés : {count} paramètres")
    print(f"Erreur maximale absolue={absolute:.3e}, relative={relative:.3e}")
    initial = model.loss(examples)
    print(f"Loss initiale : {initial:.6f}")
    for epoch in range(1, 5001):
        _, dw, db = model.loss_and_grad(examples)
        model.step(dw, db, learning_rate=0.2)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} : loss={model.loss(examples):.6f}")
    correct = 0
    for x, target in examples:
        probabilities = model.probabilities(x)
        prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        correct += prediction == target
        print(f"{x} -> {[round(p, 4) for p in probabilities]}"
              f" -> classe {prediction} (attendue {target})")
    assert correct == 4, "XOR n'a pas été appris"
    assert model.loss(examples) < initial / 10
    with tempfile.TemporaryDirectory(prefix="torch-cours-") as directory:
        path = Path(directory) / "xor.json"
        model.save(path)
        restored = MLP.load(path)
        for x, _ in examples:
            assert model.probabilities(x) == restored.probabilities(x)
    print("XOR : 4/4 ; sauvegarde/relecture : sorties identiques.")


if __name__ == "__main__":
    main()
