import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygame

from src.ai.ai_bird import AIBird, HeuristicBird, RandomBird
from src.ai.ai_config import EXPERIMENT_MAX_FRAMES, MUTATION_RATE
from src.ai.genetic_algorithm import GeneticAlgorithm
from src.ai.neural_network import NeuralNetwork
from src.ai.setup import create_config
from src.ai.simulation import Simulation

RESULTS_DIR = "results"
TEST_SEEDS = list(range(5000, 5010))


def run_evolution(config, mutation_rate, seed, generations, population_size):
    ga = GeneticAlgorithm(config, population_size, mutation_rate, seed=seed)
    birds = ga.create_population()
    for generation in range(generations):
        sim = Simulation(config, birds, seed * 1000 + generation, EXPERIMENT_MAX_FRAMES)
        while not sim.is_over():
            sim.step()
        ga.evaluate_population(birds)
        birds = ga.next_generation(birds)
    return ga.history, ga.best_genome


def test_controller(config, make_bird):
    scores = []
    for seed in TEST_SEEDS:
        birds = [make_bird(seed)]
        sim = Simulation(config, birds, seed, EXPERIMENT_MAX_FRAMES)
        while not sim.is_over():
            sim.step()
        scores.append(birds[0].pipes_passed)
    return float(np.mean(scores))


def run_all(quick):
    if quick:
        rates, seeds, generations, pop = [0.02, 0.1, 0.4], [1, 2], 15, 30
    else:
        rates, seeds, generations, pop = [0.01, 0.05, 0.1, 0.2, 0.4], [1, 2, 3, 4, 5], 40, 50

    pygame.init()
    screen = pygame.display.set_mode((288, 512))
    config = create_config(screen)
    AIBird.render = False
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {rate: [] for rate in rates}

    for rate in rates:
        for seed in seeds:
            history, genome = run_evolution(config, rate, seed, generations, pop)
            test_score = test_controller(
                config, lambda s, g=genome: AIBird(config, NeuralNetwork.from_genome(g)))
            results[rate].append({"seed": seed, "history": history, "test_pipes": test_score})
            print(f"mutation={rate:<5} seed={seed}  final best fitness="
                  f"{history['best_fitness'][-1]:.0f}  unseen-layout pipes={test_score:.1f}")

    baseline_heuristic = test_controller(config, lambda s: HeuristicBird(config))
    baseline_random = test_controller(config, lambda s: RandomBird(config, s))
    print(f"Baseline heuristic: {baseline_heuristic:.1f} pipes | random: {baseline_random:.1f} pipes")

    save_csv(results)
    make_charts(results, baseline_heuristic, baseline_random)
    print(f"Done. Table and charts are in the '{RESULTS_DIR}' folder.")


def save_csv(results):
    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mutation_rate", "seed", "final_best_fitness",
                         "final_avg_fitness", "unseen_layout_pipes"])
        for rate, runs in results.items():
            for run in runs:
                h = run["history"]
                writer.writerow([rate, run["seed"], h["best_fitness"][-1],
                                 round(h["avg_fitness"][-1], 1), run["test_pipes"]])


def mean_curve(runs, key):
    return np.mean([run["history"][key] for run in runs], axis=0)


def make_charts(results, baseline_heuristic, baseline_random):
    rates = list(results.keys())
    default_rate = MUTATION_RATE if MUTATION_RATE in results else rates[len(rates) // 2]

    # Chart 1 - best and average fitness per generation (default mutation rate)
    plt.figure(figsize=(7, 4))
    plt.plot(mean_curve(results[default_rate], "best_fitness"), label="best fitness")
    plt.plot(mean_curve(results[default_rate], "avg_fitness"), label="average fitness")
    plt.xlabel("generation"); plt.ylabel("fitness (mean over seeds)")
    plt.title(f"Fitness per generation (mutation rate {default_rate})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "chart1_fitness_by_generation.png"), dpi=150)
    plt.close()

    # Chart 2 - unseen-layout performance by mutation rate, with baselines
    plt.figure(figsize=(7, 4))
    means = [np.mean([r["test_pipes"] for r in results[rate]]) for rate in rates]
    plt.bar([str(r) for r in rates], means, color="tab:blue")
    plt.axhline(baseline_heuristic, color="green", linestyle="--", label="heuristic baseline")
    plt.axhline(baseline_random, color="red", linestyle="--", label="random baseline")
    plt.xlabel("mutation rate"); plt.ylabel("avg pipes passed (unseen layouts)")
    plt.title("Performance by mutation rate"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "chart2_pipes_by_mutation_rate.png"), dpi=150)
    plt.close()

    # Chart 3 - distribution of final performance across the repeated runs
    plt.figure(figsize=(7, 4))
    plt.boxplot([[r["test_pipes"] for r in results[rate]] for rate in rates])
    plt.xticks(range(1, len(rates) + 1), [str(r) for r in rates])
    plt.xlabel("mutation rate"); plt.ylabel("pipes passed (unseen layouts)")
    plt.title("Final performance across seeds"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "chart3_final_distribution.png"), dpi=150)
    plt.close()

    # Chart 4 - learning curves for every mutation rate
    plt.figure(figsize=(7, 4))
    for rate in rates:
        plt.plot(mean_curve(results[rate], "avg_pipes"), label=f"mutation {rate}")
    plt.xlabel("generation"); plt.ylabel("average pipes passed by the population")
    plt.title("Learning curves by mutation rate"); plt.legend()
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "chart4_learning_curves.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small, fast test run")
    run_all(parser.parse_args().quick)
