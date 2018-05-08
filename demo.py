import gym
import time

env = gym.make('MultiSnake-v0')

for i in range(100):
    obs = env.reset()
    env.render()
    time.sleep(0.5)

    for t in range(1000):
        actions = env.action_space.sample()
        obs, reward, dones, info = env.step(actions)
        env.render()
        time.sleep(0.5)

        if all(dones):
            print('episode {} finished after {} timesteps'.format(i, t))
            break
