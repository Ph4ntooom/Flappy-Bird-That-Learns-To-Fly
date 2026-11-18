import pygame

from ..utils import GameConfig, Images, Window


class _SilentSound:
    def play(self):
        pass


class SilentSounds:
    def __init__(self):
        self.die = self.hit = self.point = self.swoosh = self.wing = _SilentSound()


def create_config(screen, fps=30):
    return GameConfig(
        screen=screen,
        clock=pygame.time.Clock(),
        fps=fps,
        window=Window(288, 512),
        images=Images(),
        sounds=SilentSounds(),
    )
