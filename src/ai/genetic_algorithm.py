import numpy as np

from .ai_bird import AIBird
from .ai_config import (BASE_SEED, ELITE_COUNT, MUTATION_RATE, MUTATION_STRENGTH,
                        POPULATION_SIZE, TOURNAMENT_SIZE)
from .neural_network import NeuralNetwork


class GeneticAlgorithm:
    def __init__(self, config, population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE,
                 mutation_strength=MUTATION_STRENGTH, elite_count=ELITE_COUNT,
                 tournament_size=TOURNAMENT_SIZE, seed=BASE_SEED):
        self.config = config
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(seed)

        self.best_genome = None
        self.history = {"best_fitness": [], "avg_fitness": [],
                        "best_pipes": [], "avg_pipes": []}

    # helpers
    def _make_bird(self, brain, index):
        return AIBird(self.config, brain, colour=index)

    # generation 1
    def create_population(self):
        return [self._make_bird(NeuralNetwork(rng=self.rng), i)
                for i in range(self.population_size)]

    # evaluation
    def evaluate_population(self, birds):
        for bird in birds:
            bird.calculate_fitness()

        fitnesses = [b.fitness for b in birds]
        pipes = [b.pipes_passed for b in birds]

        self.history["best_fitness"].append(max(fitnesses))
        self.history["avg_fitness"].append(float(np.mean(fitnesses)))
        self.history["best_pipes"].append(max(pipes))
        self.history["avg_pipes"].append(float(np.mean(pipes)))

        best_bird = max(birds, key=lambda b: b.fitness)

        self.best_genome = best_bird.brain.get_genome()

    # selection
    def select_parent(self, birds):
        contenders = self.rng.choice(len(birds), self.tournament_size, replace=False)

        winner = max(contenders, key=lambda i: birds[i].fitness)

        return birds[winner]

    # crossover
    def crossover(self, genome_a, genome_b):
        take_from_a = self.rng.random(len(genome_a)) < 0.5

        return np.where(take_from_a, genome_a, genome_b)

    # mutation
    def mutate(self, genome):
        mutate_this = self.rng.random(len(genome)) < self.mutation_rate

        noise = self.rng.normal(0, self.mutation_strength, len(genome))

        return genome + mutate_this * noise

    # new population
    def next_generation(self, birds):
        ranked = sorted(birds, key=lambda b: b.fitness, reverse=True)

        # a) elitism - the best birds survive unchanged
        new_genomes = [b.brain.get_genome().copy() for b in ranked[:self.elite_count]]

        # b, c, d) fill the rest with mutated children
        while len(new_genomes) < self.population_size:
            parent_a = self.select_parent(birds)
            parent_b = self.select_parent(birds)
            child = self.crossover(parent_a.brain.get_genome(), parent_b.brain.get_genome())
            new_genomes.append(self.mutate(child))

        return [self._make_bird(NeuralNetwork.from_genome(g), i)
                for i, g in enumerate(new_genomes)]
