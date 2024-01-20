import gym
# Environment
env=gym.make("MountainCar-v0",render_mode='human')
env.reset()
print(env.state)
# obs = env.render()
# input()
# Take actions
print(env.action_space.n)
# Take X range, V range
print(env.observation_space.high)
print(env.observation_space.low)
# Render
while True:
    action=2
    #env.step(action)

    new_state, reward, done, _=env.step(action)
    print("new state={},reward={}",format(new_state,reward))
    env.render()