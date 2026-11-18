import random

from ..entities import Background, Floor, Pipes, Score


class Simulation:
    def __init__(self, config, birds, layout_seed, max_frames):
        random.seed(layout_seed)
        self.config = config
        self.birds = birds
        self.max_frames = max_frames

        self.background = Background(config)
        self.floor = Floor(config)
        self.pipes = Pipes(config)
        self.score_display = Score(config)
        self.frame = 0
        self.score = 0

    def alive_birds(self):
        return [bird for bird in self.birds if bird.alive]

    def is_over(self):
        return self.frame >= self.max_frames or not self.alive_birds()

    def step(self, render=False):
        self.frame += 1
        alive = self.alive_birds()

        # 1. collisions and scoring (same checks as Flappy.play)
        for bird in alive:
            if bird.collided(self.pipes, self.floor):
                bird.alive = False
                continue
            bird.frames_alive += 1
            for pipe in self.pipes.upper:
                if bird.crossed(pipe):
                    bird.pipes_passed += 1
                    self.score = max(self.score, bird.pipes_passed)

        # 2. every surviving bird looks at the pipes and decides whether to flap
        for bird in alive:
            if bird.alive:
                bird.think(self.pipes)

        # 3. move (and, if render=True, draw) everything
        if render:
            self.background.tick()
            self.floor.tick()
        self.pipes.tick()
        if render:
            self.score_display.score = self.score
            self.score_display.tick()
        for bird in alive:
            if bird.alive:
                bird.tick()
