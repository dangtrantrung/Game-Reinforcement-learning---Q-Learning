import gym
import numpy as np
# Environment
env=gym.make("MountainCar-v0",render_mode='human')
env.reset()
print(env.state)
c_learning_rate=0.1
c_discount_value=0.9
q_table_size=[20,20]
q_table_segment_size=(env.observation_space.high-env.observation_space.low)/q_table_size
def convert_state(real_state):

    q_state = (real_state[0] - env.observation_space.low) // q_table_segment_size
    print("real_state[0]= ",real_state[0])
    return tuple(q_state.astype(int))

q_table=np.random.uniform(low=-2,high=0,size=(q_table_size+[env.action_space.n]))
c_no_of_eps=10000
c_show_each=1000
max_ep_reward=-999
max_ep_action_list=[]
max_start_state=None

for ep in range(c_no_of_eps):
    print("Eps= ",ep)
    done=False
    current_state=convert_state(env.reset())
    ep_reward=0
    action_list=[]
    if ep%c_show_each==0:
        show_now=True
    else:
        show_now=False

    while not done:
        # get current argmax Q value of current state
        action=np.argmax(q_table[current_state])
        action_list.append(action)
        # Take next action
        next_real_state,reward,done, _ =env.step(action=action)
        ep_reward+=reward
        if show_now:
            env.render()
        if done:
                if next_real_state[0]>env.goal_position:
                    print("Reach goal at ep={}, reward={}",format(ep,ep_reward))
                    if ep_reward>max_ep_reward:
                        max_ep_reward=ep_reward
                        max_ep_action_list=action_list
                else:
                    print("Failed")
            #...
        else:
                #convert to q_state
                next_state=convert_state(next_real_state)
                # Update Q value for current_state,action
                current_q_value=q_table[current_state+(action,)]
                new_q_value=(1-c_learning_rate)*current_q_value+c_learning_rate*(reward+c_discount_value*np.max(q_table[next_state]))
                q_table[current_state+(action,)]=new_q_value
                current_state=next_state
# Print Results
print("Max reward= ",max_ep_reward)
print("Max action= ",max_ep_action_list)

env.reset()
env.state=max_start_state
for action in max_ep_action_list:
    env.step(action)
    env.render()
done=False
while not done:
    env.step(0)
    env.render()