import argparse
import asyncio

from src.flappy import Flappy


def parse_args():
    parser = argparse.ArgumentParser(description="Flappy Bird That Learns to Fly")
    parser.add_argument("--mode", choices=["play", "evolve", "replay", "heuristic"],
                        default="play")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--pop", type=int, default=50, help="population size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "play":
        asyncio.run(Flappy().start())
    else:
        from src.ai.evolution_game import EvolutionGame

        EvolutionGame(mode=args.mode, seed=args.seed, mutation_rate=args.mutation_rate,
                      population_size=args.pop).run()
