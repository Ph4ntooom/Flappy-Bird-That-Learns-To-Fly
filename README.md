# Flappy Bird That Learns to Fly
**Evolutionary learning on top of an existing Pygame Flappy Bird** - AL-2002 AI Lab, FAST-NUCES Karachi (Fall 2026)

Team: Ahad (24k-2510), Aayan Hassan (24k-2561), Amail (24k-2575) - BDS-5A

## Attribution (please read)
This repository is a **fork of [sourabhv/FlapPyBird](https://github.com/sourabhv/FlapPyBird)**
(MIT licence, Copyright Sourabh Verma). The playable game - sprites, sounds, pipes, floor, physics,
pixel-perfect collision - is **their work**, and their `LICENSE` file is kept unchanged.

**Our work** is everything in `src/ai/`, `experiments.py` and the changes to `main.py`:
a neural network, a genetic algorithm (written from scratch, no ML framework), AI-controlled birds
that *inherit* from the original `Player` class, a statistics panel, baselines and experiments.

## 1. Setup
```bash
pip install -r requirements.txt        # pygame, numpy, matplotlib   (Python 3.9+)
```
Always run commands from the repository root (the game loads `assets/...` by relative path).

## 2. How to run
| Command | What it does |
|---|---|
| `python main.py` | Play the **original** game yourself (SPACE / UP to flap) |
| `python main.py --mode evolve` | **Watch 50 birds learn** (genetic algorithm) |
| `python main.py --mode evolve --mutation-rate 0.3 --seed 7 --pop 80` | Same, other settings |
| `python main.py --mode heuristic` | Hand-written baseline bot |
| `python main.py --mode replay` | Saved best brain plays (press **S** while evolving first) |
| `python experiments.py --quick` | Fast test of the experiments (under a minute) |
| `python experiments.py` | Full mutation-rate comparison, writes `results/` (several minutes) |

**Keys in evolve mode:** SPACE pause - UP/DOWN speed (x1 to x25) - N next generation - S save best brain - ESC quit

## 3. What we changed in the fork
| File | Status | Purpose |
|---|---|---|
| `main.py` | **modified** | Command-line options; `--mode play` runs the original game exactly as before |
| `src/ai/ai_config.py` | new | All AI settings (population, mutation rate, fitness bonus...) |
| `src/ai/neural_network.py` | new | `NeuralNetwork` - 4-6-1 network, genome get/set |
| `src/ai/ai_bird.py` | new | `AIBird(Player)`, `HeuristicBird(AIBird)`, `RandomBird(AIBird)` |
| `src/ai/genetic_algorithm.py` | new | `GeneticAlgorithm` - evaluation, tournament selection, crossover, mutation, elitism |
| `src/ai/simulation.py` | new | `Simulation` - one round of the ORIGINAL game with many birds |
| `src/ai/evolution_game.py` | new | `EvolutionGame` - window, stats panel, keyboard |
| `src/ai/setup.py` | new | Builds the game config with silent sounds |
| `experiments.py` | new | Headless mutation-rate comparison, charts, CSV |
| `requirements.txt`, `.gitignore`, `README.md` | new / edited | |
| `.github/workflows/deploy.yaml` | **deleted** | Upstream web-deploy job, not needed for our project |

Everything else under `src/` (entities, utils) and `assets/` is **unchanged upstream code**.

## 4. How it works (for the evaluation)
**Inheritance:** `AIBird` extends the original `Player`, so physics, wing animation, flapping and
collision are reused. It *overrides* `update_image()` and `draw_player()`. `HeuristicBird` and
`RandomBird` extend `AIBird` and *override* `decide()`.

**Sensors (4 inputs, scaled):** bird height, vertical velocity, horizontal distance to the next pipe,
vertical offset of the gap centre from the bird.
**Action:** network output > 0 means flap.
**Fitness:** `frames survived + 100 x pipes passed`.
**Genome:** all 49 weights/biases of the network flattened into one array.

Each generation:
1. All birds play the same random pipe layout at the same time (`Simulation`).
2. Fitness is computed.
3. **Elitism:** the best 2 birds are copied unchanged.
4. **Selection:** tournament of 3 random birds, the fittest becomes a parent.
5. **Crossover:** every weight comes from parent A or B (coin flip - uniform crossover).
6. **Mutation:** each weight has a `mutation_rate` chance of Gaussian noise (std 0.3).

Every generation gets a new layout (derived from the seed), so a bird cannot memorise one layout.
The original pipes use Python's global `random`, so `Simulation` calls `random.seed(layout_seed)`
first - same seed, same pipes, reproducible experiments.

**PEAS:** Performance = fitness - Environment = the original Flappy Bird game - Actuators = flap / no flap -
Sensors = the 4 features above.

## 5. Experiments (`results/`)
`summary.csv` plus 4 charts: (1) best/avg fitness per generation, (2) pipes passed by mutation rate vs.
baselines, (3) final-performance distribution across seeds, (4) learning curves per mutation rate.
The final best bird of every run is tested on 10 pipe layouts never seen during evolution.

## 6. Limitations
- Results differ a lot between seeds; some runs do not converge within the generation budget.
- Experiment rounds are capped at 3000 frames (~75 pipes) to keep them fast; the GUI cap is 100000 frames.
- Five seeds per setting: the mutation-rate comparison is indicative, not statistically conclusive.
- Dead birds simply disappear (the original crash animation is only used in human play).
