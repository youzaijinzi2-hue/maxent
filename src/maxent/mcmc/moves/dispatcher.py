import random


class MoveDispatcher:
    """
    Mixture of proposal kernels:
        K = sum_m p_m K_m
    """
    def __init__(self, moves: dict, probs: dict):
        self.moves = moves
        self.probs = probs
        self.names = list(moves.keys())
        self.weights = [probs[k] for k in self.names]

    def sample(self):
        name = random.choices(self.names, weights=self.weights, k=1)[0]
        return name, self.moves[name]