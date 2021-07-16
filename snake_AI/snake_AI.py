import random
from collections import deque, namedtuple
from enum import Enum

import pygame

# from snake_model import Linear_QNet, Training


pygame.init()
font = pygame.font.SysFont('arial', 25)
Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class SnakeGame():
    def __init__(self, width = 32, height = 18, blocksize = 60, init_snake_length = 3, gamespeed = 10, food_gain = 1):
        # init user set parameters
        self.blocksize = blocksize
        self.width = width
        self.height = height

        self.gamespeed = gamespeed
        self.food_gain = food_gain
        self.init_snake_length = init_snake_length

        # init display
        self.show_UI = 1
        if self.show_UI:
            self.display = pygame.display.set_mode((self.width * self.blocksize, self.height * self.blocksize))
            pygame.display.set_caption('High IQ Snake!!!')
            self.clock = pygame.time.Clock()
        
        self.restart()


    def restart(self):
        # init 
        self.score = 0
        self.game_over = 0
        self.direction = Direction.RIGHT
        self.food = None
        self.snake_buffer = self.init_snake_length

        self.head = self.init_head()
        self.snake = self.init_snake()
        self.place_food()


    def place_food(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        self.food = Point(x, y)

        while self.food in self.snake:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.food = Point(x, y)

        return self.food
    

    def init_head(self):
        x = random.randint(self.width // 4, self.width // 4 * 3)
        y = random.randint(self.height // 4, self.height // 4 * 3)
        init_point = Point(x, y)

        return init_point
    

    def init_snake(self):
        snake = deque()
        snake.appendleft(Point(self.head.x, self.head.y))

        return snake


    def update_snake_and_food(self):
        if self.direction == Direction.UP:
            self.head = Point(self.head.x, self.head.y - 1)
            
        if self.direction == Direction.DOWN:
            self.head = Point(self.head.x, self.head.y + 1)

        if self.direction == Direction.LEFT:
            self.head = Point(self.head.x - 1, self.head.y)

        if self.direction == Direction.RIGHT:
            self.head = Point(self.head.x + 1, self.head.y)


        self.snake.appendleft(self.head)

        if self.head == self.food:
            self.snake_buffer = self.snake_buffer + self.food_gain
            self.score += 1
            self.place_food()
            self.snake_buffer -= 1
        elif self.snake_buffer != 0:
            self.snake_buffer -= 1
        else:
            self.snake.pop()
            

    
    def play_step(self):
        for event in pygame.event.get():
        
            if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                    
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.direction = Direction.UP

                if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.direction = Direction.DOWN

                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    self.direction = Direction.LEFT

                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    self.direction = Direction.RIGHT
        
        self.update_snake_and_food()

        if self.is_collision(): 
            self.game_over = 1
            return self.game_over, self.score

        if self.show_UI:
            self.update_UI()

        self.clock.tick(self.gamespeed)

        return self.game_over, self.score
    

    def is_collision(self):
        if self.head.x > self.width - 1 or self.head.x < 0 or self.head.y > self.height - 1 or self.head.y < 0:
            return True
        
        for i in range(1, len(self.snake)):
            if self.head == self.snake[i]: 
                return True
            
            elif i == len(self.snake):
                return False


    def update_UI(self):
        self.display.fill(BLACK)

        for i, pt in enumerate(self.snake):
            if i == 0: pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x * self.blocksize, pt.y * self.blocksize, self.blocksize, self.blocksize))
            else: pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * self.blocksize, pt.y * self.blocksize, self.blocksize, self.blocksize))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x * self.blocksize, self.food.y * self.blocksize, self.blocksize, self.blocksize))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        return



def main():
    game = SnakeGame()
    game_over = 1

    while True:
        game_over, score = game.play_step()

        if game_over: 
            game.restart()

    print(score)


if __name__ == '__main__':
    main()





























