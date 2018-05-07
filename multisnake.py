import gym
import copy
import random
import numpy as np
from gym import spaces, logger
from collections import deque
from gym.envs.classic_control import rendering

'''
This environment is built based on the file from
https://github.com/SeanBae/gym-snake/blob/master/gym_snake/envs/snake_env.py
'''

class SnakeAction:
    LEFT = 0
    RIGHT = 1
    FORWARD = 2

class SnakeCellState:
    EMPTY = 0
    WALL = 1
    FRUIT = 2
    BODY = 3
    SNAKE_FRUIT = 4

class SnakeReward:
    ALIVE = 0
    FRUIT = 5
    SNAKE_FRUIT = 15
    DEAD = -20

class SnakeGame:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.n_snakes = 4
        init_snake_length = 4

        while True:
            all_success = []
            self.snakes = [deque() for _ in range(self.n_snakes)]
            self.snakes_dead = [False for _ in range(self.n_snakes)]
            self.snakes_become_fruits = [False for _ in range(self.n_snakes)]

            self.empty_cells = {(x, y) for x in range(width) for y in range(height)}
            head_random_area = {(x, y) for x in range(width//4, 3*width//4) \
                                for y in range(height//4, 3*height//4)}
            self.fruits = set()
            self.snake_fruits = set()

            for snake_num in range(self.n_snakes):
                while True:
                    head = random.sample(head_random_area, 1)[0]
                    if self.cell_state(head) == SnakeCellState.EMPTY:
                        break
    
                self.add_to_head(head,snake_num)
                for _ in range(init_snake_length - 1):
                    success = self.add_to_tail(snake_num)
                    all_success.append(success)
            if all(all_success):
                break

        self.n_fruits = 10
        for _ in range(self.n_fruits):
            self.generate_fruit()

    def cell_state(self, cell):
        # return one cell's state

        if cell in self.snake_fruits:
            return SnakeCellState.SNAKE_FRUIT

        if cell in self.fruits:
            return SnakeCellState.FRUIT

        if cell in self.empty_cells:
            return SnakeCellState.EMPTY

        for i in range(self.n_snakes):
            if not self.snakes_dead[i]:
                if cell in self.snakes[i]:
                    return SnakeCellState.BODY

        x, y = cell
        assert x<0 or y<0 or x>=self.width or y>=self.height
        return SnakeCellState.WALL
    '''
    Fruit Functions
    '''
    def can_generate_fruit(self):
        # determine if there is empty space to generate a fruit
        return len(self.empty_cells) > 0

    def generate_fruit(self):
        new_fruit = random.sample(self.empty_cells, 1)[0]
        assert self.cell_state(new_fruit) == SnakeCellState.EMPTY

        self.empty_cells.remove(new_fruit)
        self.fruits.add(new_fruit)

    def generate_fruit_at_death(self, new_fruit):
        self.snake_fruits.add(new_fruit)

    '''
    Snake Functions
    '''
    def add_to_head(self, cell, number):
        self.snakes[number].appendleft(cell)
        if cell in self.empty_cells:
            self.empty_cells.remove(cell)

        if cell in self.fruits:
            self.fruits.remove(cell)

        if cell in self.snake_fruits:
            self.snake_fruits.remove(cell)

    def add_to_tail(self, number):
        last_tail = self.snakes[number][-1]
        x, y = last_tail
        neighbor_cells = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        avail_cells = []
        
        for cell in neighbor_cells:
            if self.cell_state(cell) == SnakeCellState.EMPTY:
                avail_cells.append(cell)
        
        if len(avail_cells) == 0:
            return False
        new_tail = random.sample(avail_cells, 1)[0]
        self.snakes[number].append(new_tail)
        self.empty_cells.remove(new_tail)
        return True

    def head(self, number):
        # return the position of the head
        return self.snakes[number][0]

    def tail(self, number):
        # return the position of the tail
        return self.snakes[number][-1]

    def remove_tail(self, number):
        tail = self.snakes[number].pop()
        if tail not in self.snakes[number]:
            self.empty_cells.add(tail)
        return tail

    def next_head(self, action, number):
        h_x, h_y = self.head(number)
        h2_x, h2_y = self.snakes[number][1]
        
        if h2_x == h_x - 1:
            if action == SnakeAction.LEFT:
                return (h_x, h_y + 1)
            if action == SnakeAction.RIGHT:
                return (h_x, h_y - 1)
            if action == SnakeAction.FORWARD:
                return (h_x + 1, h_y)
        elif h2_x == h_x + 1:
            if action == SnakeAction.LEFT:
                return (h_x, h_y - 1)
            if action == SnakeAction.RIGHT:
                return (h_x, h_y + 1)
            if action == SnakeAction.FORWARD:
                return (h_x - 1, h_y)
        elif h2_y == h_y + 1:
            if action == SnakeAction.LEFT:
                return (h_x + 1, h_y)
            if action == SnakeAction.RIGHT:
                return (h_x - 1, h_y)
            if action == SnakeAction.FORWARD:
                return (h_x, h_y - 1)
        else:
            if action == SnakeAction.LEFT:
                return (h_x - 1, h_y)
            if action == SnakeAction.RIGHT:
                return (h_x + 1, h_y)
            if action == SnakeAction.FORWARD:
                return (h_x, h_y + 1)

    def delete_death(self):
        for i in range(self.n_snakes):
            if self.snakes_dead[i]:
                if not self.snakes_become_fruits[i]:
                    for cell in set(self.snakes[i]):
                        self.generate_fruit_at_death(cell)
                    self.snakes_become_fruits[i] = True

    def step(self, actions):
        rewards = np.zeros((self.n_snakes,))
        next_heads = []
        next_states = [-1 for _ in range(self.n_snakes)]

        death_list = copy.deepcopy(self.snakes_dead)
        process_list = copy.deepcopy(self.snakes_dead)

        for num in range(self.n_snakes):
            if self.snakes_dead[num]:
                next_heads.append((-10,-10))
                continue
            # record all the next heads and next states
            action = actions[num]
            next_head = self.next_head(action, num)
            
            # check if co-next_head
            if next_head in next_heads:
                rewards[num] = SnakeReward.DEAD
                death_list[num] = True
                process_list[num] = True

            next_heads.append(next_head)
            next_states[num] = self.cell_state(next_head)

        for num in range(self.n_snakes):
            if process_list[num]:
                continue
            # deal with the wall first
            if next_states[num] == SnakeCellState.WALL:
                rewards[num] = SnakeReward.DEAD
                death_list[num] = True
                process_list[num] = True

            # deal with the self_eat second
            if next_states[num] == SnakeCellState.BODY:
                if next_heads[num] == self.tail(num):
                    self.remove_tail(num)
                    self.add_to_head(next_heads[num], num)
                    rewards[num] = SnakeReward.ALIVE
                    process_list[num] = True

            # deal with the fruit_eat third
            if next_states[num] == SnakeCellState.FRUIT:
                self.add_to_head(next_heads[num], num)
                rewards[num] = SnakeReward.FRUIT
                process_list[num] = True
            if next_states[num] == SnakeCellState.SNAKE_FRUIT:
                self.add_to_head(next_heads[num], num)
                rewards[num] = SnakeReward.SNAKE_FRUIT
                process_list[num] = True

        #update the next states
        for num in range(self.n_snakes):
            if process_list[num]:
                continue
            next_states[num] = self.cell_state(next_heads[num])

        for _ in range(self.n_snakes):
            for num in range(self.n_snakes):
                if process_list[num]:
                    continue
                # deal with the death first
                if next_states[num] == SnakeCellState.BODY:
                    dead = False
                    have = False
                    for tail_num in range(self.n_snakes):
                        if self.snakes_dead[tail_num]:
                            continue
                        if next_heads[num] in self.snakes[tail_num]:
                            have = True
                            if process_list[tail_num] or\
                                next_heads[num] != self.tail(tail_num):
                                dead =  True
                    assert have == True
                    if dead:
                        rewards[num] = SnakeReward.DEAD
                        death_list[num] = True
                        process_list[num] = True

        tail_removed = []
        for num in range(self.n_snakes):
            if process_list[num]:
                tail_removed.append((-5,-5))
                continue
            tail = self.remove_tail(num)
            tail_removed.append(tail)

        #update the next states
        for num in range(self.n_snakes):
            if process_list[num]:
                continue
            next_states[num] = self.cell_state(next_heads[num])

        no_need = True
        for num in range(self.n_snakes):
            if process_list[num]:
                continue
            # deal with the alive
            if next_states[num] == SnakeCellState.EMPTY:
                self.add_to_head(next_heads[num], num)
                rewards[num] = SnakeReward.ALIVE
                process_list[num] = True
            else:
                assert no_need == False
                self.snakes[num].append(tail_removed[num])
                if tail_removed[num] in self.empty_cells:
                    self.empty_cells.remove(tail_removed[num])
                rewards[num] = SnakeReward.DEAD
                death_list[num] = True
                process_list[num] = True

        assert all(process_list) == True
        self.snakes_dead = copy.deepcopy(death_list)
        self.delete_death()
        assert self.snakes_become_fruits == self.snakes_dead

        while True:
            if len(self.fruits) >= self.n_fruits or (not self.can_generate_fruit()):
                break
            self.generate_fruit()

        return rewards

    def get_obs(self):
        # return the nparray that represent the state
        all_obs = []
        wall_cell = 50
        sbody_cell = 140
        shead_cell = 180
        fruit_cell = 255
        
        opbody_cell = 80
        opbhead_cell = 120
        opfruit_cell = 230
        
        for num in range(self.n_snakes):
            obs = np.zeros((1, (self.width + 2), (self.height + 2)), dtype = np.uint8)
            
            for x in range(self.width + 2):
                for y in range(self.height + 2):
                    if x == 0 or y == 0 or x == self.width+1 or y == self.height+1:
                         obs[0, x, y] = wall_cell
    
            for i in range(self.n_snakes):
                if not self.snakes_dead[i]:
                    for k in reversed(range(len(self.snakes[i]))):
                        (x, y) = self.snakes[i][k]
                        if i == num:
                            if k == 0:
                                obs[0, x+1, y+1] = shead_cell
                            else:
                                obs[0, x+1, y+1] = sbody_cell
                        else:
                            if k == 0:
                                obs[0, x+1, y+1] = opbhead_cell
                            else:
                                obs[0, x+1, y+1] = opbody_cell

            for each_fruit in self.fruits:
                x, y = each_fruit
                obs[0, x+1, y+1] = fruit_cell
    
            for each_fruit in self.snake_fruits:
                x, y = each_fruit
                obs[0, x+1, y+1] = opfruit_cell
    
            obs = obs.transpose([1, 2, 0])
            #obs = obs/128 - 1
            
            h_x, h_y = self.head(num)
            h2_x, h2_y = self.snakes[num][1]
            
            if h2_x == h_x - 1:
                all_obs.append(np.rot90(np.rot90(obs)))
            elif h2_x == h_x + 1:
                all_obs.append(obs)
            elif h2_y == h_y + 1:
                all_obs.append(np.rot90(np.rot90(np.rot90(obs))))
            else:
                all_obs.append(np.rot90(obs))

        return all_obs


class MultiSnakeEnv(gym.Env):
    metadata= {'render.modes': ['human'], 'video.frames_per_second': 2}

    def __init__(self):
        self.width = 18
        self.height = 18
        self.game = SnakeGame(self.width, self.height)
        
        self.action_space = spaces.MultiDiscrete([3] * self.game.n_snakes)
        self.observation_space = spaces.Box(low = 0, high = 255, 
                                            shape = (self.width+2, self.height+2, 1))
        self.viewer = None
        self.game_over = True

    def step(self, action):
        if self.game_over:
            logger.warn("You are calling step before resetting the game or \
                        when the game is over!")
            return self.game.get_obs(), 0, all(self.game.snakes_dead), {}
        
        rewards = self.game.step(action)
        done = self.game.snakes_dead
        observation = self.game.get_obs()
        info = {}

        if all(self.game.snakes_dead):
            self.game_over = True

        return (observation, rewards, done, info)

    def reset(self):
        self.game = SnakeGame(self.width, self.height)
        observation = self.game.get_obs()
        self.game_over = False
        
        return observation

    def render(self, mode='human', close=False):
        width = height = 800
        width_scaling_factor = width / self.width
        height_scaling_factor = height / self.height

        if self.viewer is None:
            self.viewer = rendering.Viewer(width, height)
        
        for i in range(self.game.n_snakes):
            if not self.game.snakes_dead[i]:
                for k in reversed(range(len(self.game.snakes[i]))):
                    (x, y) = self.game.snakes[i][k]
                    l, r, t, b = ((x+0.05) * width_scaling_factor,
                                  (x+0.95) * width_scaling_factor, 
                                  (y+0.05) * height_scaling_factor, 
                                  (y+0.95) * height_scaling_factor)
                    square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                    if k == 0:
                        square.set_color(0, 0, 0)
                    else:
                        if i == 0:
                            square.set_color(0, 0, 255)
                        elif i == 1:
                            square.set_color(0, 255, 0)
                        elif i == 2:
                            square.set_color(255, 0, 255)
                        else:
                            square.set_color(0, 255, 255)
                    self.viewer.add_onetime(square)

        for each_fruit in self.game.fruits:
            x, y = each_fruit
            l, r, t, b = ((x+0.05) * width_scaling_factor, 
                          (x+0.95) * width_scaling_factor, 
                          (y+0.05) * height_scaling_factor, 
                          (y+0.95) * height_scaling_factor)
            square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            square.set_color(255, 0, 0)
            self.viewer.add_onetime(square)
        
        for each_fruit in self.game.snake_fruits:
            x, y = each_fruit
            l, r, t, b = ((x+0.05) * width_scaling_factor, 
                          (x+0.95) * width_scaling_factor, 
                          (y+0.05) * height_scaling_factor, 
                          (y+0.95) * height_scaling_factor)
            square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            square.set_color(255, 255, 0)
            self.viewer.add_onetime(square)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        pass

    def seed(self):
        pass
