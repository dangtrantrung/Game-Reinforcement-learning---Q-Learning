import gym
# Environment
env=gym.make("MountainCar-v0",render_mode="rgb_array")
env.reset()
print(env.state)
env.render()
input()