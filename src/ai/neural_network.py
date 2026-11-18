import numpy as np

from .ai_config import HIDDEN_SIZE, INPUT_SIZE


class NeuralNetwork:
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, rng=None):
        rng = rng if rng is not None else np.random.default_rng()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Random starting weights between -1 and 1
        self.w1 = rng.uniform(-1, 1, (input_size, hidden_size))
        self.b1 = rng.uniform(-1, 1, hidden_size)
        self.w2 = rng.uniform(-1, 1, hidden_size)
        self.b2 = rng.uniform(-1, 1)

    def forward(self, inputs):
        hidden = np.tanh(inputs @ self.w1 + self.b1)

        return float(np.tanh(hidden @ self.w2 + self.b2))

    #  genome <-> weights
    def get_genome(self):
        return np.concatenate([self.w1.ravel(), self.b1, self.w2, [self.b2]])

    def set_genome(self, genome):
        n_w1 = self.input_size * self.hidden_size
        n_b1 = self.hidden_size

        self.w1 = genome[:n_w1].reshape(self.input_size, self.hidden_size)
        self.b1 = genome[n_w1:n_w1 + n_b1].copy()
        self.w2 = genome[n_w1 + n_b1:n_w1 + n_b1 + self.hidden_size].copy()
        self.b2 = float(genome[-1])

    @classmethod
    def from_genome(cls, genome):
        brain = cls()
        
        brain.set_genome(np.array(genome, dtype=float))

        return brain
