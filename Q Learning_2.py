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
    q_state=(real_state-env.observation_space.low) // q_table_size
    return tuple(q_state.astype(np.int))

q_table=np.random.uniform(low=-2,high=0,size=(q_table_size+[env.action_space.n]))
c_no_of_eps=10000
for ep in range(c_no_of_eps):
    print("Eps= ",ep)
    done=False
    current_state=convert_state(env.reset())

    while not done:
        # get current argmax Q value of current state
        action=np.argmax(q_table[current_state])
        # Take next action
        next_real_state,reward,done, _ =env.step(action=action)
        if done:
                if next_real_state[0]>env.goal_position:
                    print("Reach goal at ep=",ep)
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