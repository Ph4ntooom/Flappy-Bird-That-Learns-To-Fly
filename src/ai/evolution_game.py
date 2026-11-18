import os
import sys

import numpy as np
import pygame

from .ai_bird import AIBird, HeuristicBird
from .ai_config import (BASE_SEED, BEST_BRAIN_FILE, GUI_MAX_FRAMES, MUTATION_RATE,
                        PANEL_WIDTH, POPULATION_SIZE)
from .genetic_algorithm import GeneticAlgorithm
from .neural_network import NeuralNetwork
from .setup import create_config
from .simulation import Simulation

GAME_WIDTH, GAME_HEIGHT = 288, 512
SPEED_LEVELS = [1, 2, 5, 10, 25]

PANEL_BG = (30, 33, 48)
WHITE = (240, 240, 240)
GREEN = (90, 220, 120)
BLUE = (100, 170, 255)
GREY = (150, 150, 165)


class EvolutionGame:
    def __init__(self, mode="evolve", seed=BASE_SEED, mutation_rate=MUTATION_RATE,
                 population_size=POPULATION_SIZE):
        pygame.init()
        pygame.display.set_caption("Flappy Bird That Learns to Fly")
        self.display = pygame.display.set_mode((GAME_WIDTH + PANEL_WIDTH, GAME_HEIGHT))
        game_surface = self.display.subsurface((0, 0, GAME_WIDTH, GAME_HEIGHT))
        self.config = create_config(game_surface)      # the original game draws onto game_surface

        self.font = pygame.font.SysFont(None, 26)
        self.small_font = pygame.font.SysFont(None, 21)
        self.big_font = pygame.font.SysFont(None, 34)

        self.mode = mode
        self.seed = seed
        self.ga = GeneticAlgorithm(self.config, population_size, mutation_rate, seed=seed)

        self.generation = 1
        self.speed_index = 0
        self.paused = False
        self.best_pipes_ever = 0
        self.message = ""

        self.birds = self._create_birds()
        self._start_round()


    def _create_birds(self):
        if self.mode == "evolve":
            return self.ga.create_population()
        if self.mode == "replay":
            if not os.path.exists(BEST_BRAIN_FILE):
                sys.exit(f"No saved brain found ({BEST_BRAIN_FILE}). "
                         "Run evolve mode and press S first.")
            return [AIBird(self.config, NeuralNetwork.from_genome(np.load(BEST_BRAIN_FILE)))]
        return [HeuristicBird(self.config)]

    def _start_round(self):
        layout_seed = self.seed * 1000 + self.generation
        self.sim = Simulation(self.config, self.birds, layout_seed, GUI_MAX_FRAMES)

    def _end_round(self):
        self.best_pipes_ever = max(self.best_pipes_ever, self.sim.score)
        if self.mode == "evolve":
            self.ga.evaluate_population(self.birds)
            self.birds = self.ga.next_generation(self.birds)
        else:
            self.birds = self._create_birds()
        self.generation += 1
        self._start_round()


    def run(self):
        while True:
            self._handle_events()
            if not self.paused:
                steps = SPEED_LEVELS[self.speed_index]
                for i in range(steps):
                    AIBird.render = (i == steps - 1)
                    self.sim.step(render=AIBird.render)
                    if self.sim.is_over():
                        self._end_round()
            self._draw_panel()
            pygame.display.update()
            self.config.tick()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_UP:
                self.speed_index = min(self.speed_index + 1, len(SPEED_LEVELS) - 1)
            elif event.key == pygame.K_DOWN:
                self.speed_index = max(self.speed_index - 1, 0)
            elif event.key == pygame.K_n:
                for bird in self.birds:
                    bird.alive = False
            elif event.key == pygame.K_s and self.ga.best_genome is not None:
                np.save(BEST_BRAIN_FILE, self.ga.best_genome)
                self.message = f"Saved best brain to {BEST_BRAIN_FILE}"


    def _text(self, text, x, y, font=None, colour=WHITE):
        self.display.blit((font or self.font).render(text, True, colour), (x, y))

    def _draw_panel(self):
        pygame.draw.rect(self.display, PANEL_BG, (GAME_WIDTH, 0, PANEL_WIDTH, GAME_HEIGHT))
        x = GAME_WIDTH + 15
        history = self.ga.history
        alive = len(self.sim.alive_birds())

        self._text("EVOLUTION" if self.mode == "evolve" else self.mode.upper(),
                   x, 15, self.big_font, BLUE)
        self._text(f"Generation:  {self.generation}", x, 55)
        self._text(f"Alive:  {alive} / {len(self.birds)}", x, 82)
        self._text(f"Pipes passed:  {self.sim.score}", x, 109)
        self._text(f"Best ever:  {self.best_pipes_ever} pipes", x, 136)

        if self.mode == "evolve":
            self._text(f"Mutation rate:  {self.ga.mutation_rate}", x, 170, self.small_font, GREY)
            if history["best_fitness"]:
                self._text(f"Last gen best fitness:  {history['best_fitness'][-1]:.0f}",
                           x, 195, self.small_font)
                self._text(f"Last gen avg fitness:  {history['avg_fitness'][-1]:.0f}",
                           x, 217, self.small_font)
            self._draw_chart(x, 245, PANEL_WIDTH - 30, 130)

        speed = f"Speed:  x{SPEED_LEVELS[self.speed_index]}" + ("   (PAUSED)" if self.paused else "")
        self._text(speed, x, 392, self.small_font, GREEN)
        if self.message:
            self._text(self.message, x, 412, self.small_font, GREEN)

        controls = ["SPACE  pause", "UP/DOWN  speed", "N  next generation",
                    "S  save best brain", "ESC  quit"]
        for i, line in enumerate(controls):
            self._text(line, x, 424 + i * 17, self.small_font, GREY)

    def _draw_chart(self, x, y, w, h):
        pygame.draw.rect(self.display, (20, 22, 32), (x, y, w, h))
        pygame.draw.rect(self.display, GREY, (x, y, w, h), 1)
        best = self.ga.history["best_fitness"]
        avg = self.ga.history["avg_fitness"]
        self._text("fitness per generation", x + 4, y + 3, self.small_font, GREY)
        if len(best) < 2:
            return
        top = max(best) or 1
        for series, colour in ((best, GREEN), (avg, BLUE)):
            points = [(x + i * w / (len(series) - 1), y + h - 4 - (value / top) * (h - 24))
                      for i, value in enumerate(series)]
            pygame.draw.lines(self.display, colour, False, points, 2)
