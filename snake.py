import pygame
import neat
import time
import os
import random
pygame.font.init()
WIN_WIDTH = 500
WIN_HEIGHT = 500


STAT_FONT = pygame.font.SysFont("comicsans", 50)


SNAKE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "snake.png")))
POINT = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "point.png")))
BG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "black.png")))

class Snake:
    IMG = SNAKE_IMG
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = 1
        self.pos = [[x,y]]
        self.last_dir = "up"
        self.moves = 0

    def size_aug(self):
        self.length += 1
    
    def up(self):
        if self.last_dir != "down":
            self.last_dir = "up"
    
    def down(self):
        if self.last_dir != "up":
            self.last_dir = "down"

    def left(self):
        if self.last_dir != "right":
            self.last_dir = "left"
    
    def right(self):
        if self.last_dir != "left":
            self.last_dir = "right"

    def move(self):
        if self.last_dir == "up":
            self.y -= 20
        elif self.last_dir == "down":
            self.y += 20
        elif self.last_dir == "left":
            self.x -= 20
        else:
            self.x += 20
        if self.moves > 10:
            self.moves = 0
    
        
        self.pos += [[self.x,self.y]]
    
    def remove_tail(self,win):
        win.blit(BG , (self.pos[0][0], self.pos[0][1]))
        self.pos = self.pos[1:]

    
    def draw(self, win):
        win.blit(self.IMG, (self.x,self.y))


class Point:
    IMG = POINT

    def __init__(self):
        self.x = random.randrange(1,48)*10
        self.y = random.randrange(1,48)*10
    
    def draw(self,win):
        win.blit(self.IMG, (self.x, self.y))

    def remove(self, win):
        win.blit(BG, (self.x,self.y))

    def pos(self):
        return [self.x, self.y]
        

def draw_window(win, snakes, score):
    win.blit(BG, (0,0))

    text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    
    for snake in snakes:
        snake.draw(win)
    
    pygame.display.update()


def main(genomes, config):

    nets = []
    ge= []
    snakes = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snakes.append(Snake(WIN_WIDTH//2, WIN_HEIGHT//2))
        g.fitness = 0
        ge.append(g)


    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0
    point = Point()

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        point.draw(win)
        if len(snakes)==0:
            run= False
            break
        for index, snake in enumerate(snakes):
            if snake.moves >=10:
                ge[index].fitness -= 0.1
                print("move>10", ge[index].fitness)
            snake.move()
            ge[index].fitness += 0.1

            output = nets[index].activate((snake.pos[-1][0], snake.pos[-1][1], abs(snake.x-point.pos()[0]), abs(snake.y-point.pos()[1])))
            if output[0] == max(output):
                snake.up()
            elif output[1] == max(output):
                snake.down()
            elif output[2] == max(output):
                snake.left()
            else:
                snake.right()
            if snake.pos[-1] in snake.pos[:-1]:
                ge[index].fitness -= 5
                snakes.pop(index)
                nets.pop(index)
                ge.pop(index)
            for i in snake.pos:
                if point.pos()[0]- i[0]<=20 and point.pos()[0]- i[0]>=0 and point.pos()[1]- i[1]>= 0 and point.pos()[1]- i[1]<=20:
                    point.remove(win)
                    point = Point()
                    score += 1
                    snake.size_aug()
                    ge[index].fitness += 10
            if snake.length < len(snake.pos):
                snake.remove_tail(win)
        for index, snake in enumerate(snakes):
            if snake.pos[-1][0] < 0 or snake.pos[-1][0] > 480 or snake.pos[-1][1] < 0 or snake.pos[-1][1] > 480:
                ge[index].fitness -= 5
                snakes.pop(index)
                nets.pop(index)
                ge.pop(index)


        draw_window(win, snakes, score)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-neat.txt")
    run(config_path)