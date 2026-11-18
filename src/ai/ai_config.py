# neural network

# bird height, velocity, distance to pipe, gap offset
INPUT_SIZE = 4
# neurons in the hidden layer
HIDDEN_SIZE = 6


# genetic algorithm

POPULATION_SIZE = 50
# best birds copied unchanged into the next generation
ELITE_COUNT = 2
# birds that compete when choosing a parent
TOURNAMENT_SIZE = 3
# probability that ONE weight gets mutated
MUTATION_RATE = 0.10
# size (std-dev) of a mutation
MUTATION_STRENGTH = 0.30
# fitness reward for every pipe passed
PIPE_BONUS = 100
# makes runs reproducible
BASE_SEED = 42


# simulation

# a round in the GUI stops after this many frames (press N to skip)
GUI_MAX_FRAMES = 100000
# round cap for the headless experiments (keeps them fast)
EXPERIMENT_MAX_FRAMES = 3000


# GUI / files

# width of the statistics panel next to the game
PANEL_WIDTH = 260
# saved with the S key, loaded by replay mode
BEST_BRAIN_FILE = "best_brain.npy"