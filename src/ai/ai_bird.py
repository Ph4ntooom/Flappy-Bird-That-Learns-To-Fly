import numpy as np
import pygame

from ..entities import Player, PlayerMode
from ..utils.constants import PLAYERS
from .ai_config import PIPE_BONUS

_sprite_cache = None


def load_bird_sprites():
    global _sprite_cache
    if _sprite_cache is None:
        _sprite_cache = [tuple(pygame.image.load(path).convert_alpha() for path in colour)
                         for colour in PLAYERS]
    return _sprite_cache


class AIBird(Player):
    render = True

    def __init__(self, config, brain=None, colour=0):
        super().__init__(config)
        self.brain = brain
        self.sprites = load_bird_sprites()[colour % 3]
        self.image = self.sprites[0]
        self.alive = True
        self.frames_alive = 0
        self.pipes_passed = 0
        self.fitness = 0.0
        self.set_mode(PlayerMode.NORMAL)

    # sensors
    def next_pipe_pair(self, pipes):
        for upper, lower in zip(pipes.upper, pipes.lower):
            if upper.x + upper.w > self.x:
                return upper, lower
            
        return pipes.upper[-1], pipes.lower[-1]

    def get_inputs(self, pipes):
        upper, lower = self.next_pipe_pair(pipes)
        gap_center = (upper.y + upper.h + lower.y) / 2

        view_height = self.config.window.viewport_height

        return np.array([
            # height
            self.cy / view_height,
            # speed
            self.vel_y / self.max_vel_y,
            # distance to next pipe
            (upper.x - self.x) / self.config.window.width,
            # gap center relative
            (gap_center - self.cy) / view_height,
        ])

    # decision
    def decide(self, inputs):
        return self.brain.forward(inputs) > 0

    def think(self, pipes):
        if self.decide(self.get_inputs(pipes)):
            self.flap()

    # overridden Player methods
    def update_image(self):
        self.frame += 1
        if self.frame % 5 == 0:
            self.img_idx = next(self.img_gen)
            self.image = self.sprites[self.img_idx]

    def draw_player(self):
        if AIBird.render:
            super().draw_player()

    # fitness
    def calculate_fitness(self):
        
        self.fitness = self.frames_alive + PIPE_BONUS * self.pipes_passed

        return self.fitness


class HeuristicBird(AIBird):

    FLAP_BELOW_PIXELS = 10

    def decide(self, inputs):
        # decision to flap if gap is below the bird
        gap_offset = inputs[3] * self.config.window.viewport_height
        return gap_offset < -self.FLAP_BELOW_PIXELS


class RandomBird(AIBird):

    def __init__(self, config, seed=0):
        super().__init__(config)
        self.rng = np.random.default_rng(seed)

    def decide(self, inputs):
        return self.rng.random() < 0.10
