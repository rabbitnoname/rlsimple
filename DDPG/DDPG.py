import sys
import os

currentDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.abspath(os.path.join(currentDir,os.pardir)) # this will return parent directory.
sys.path.insert(0,parentDir)

import numpy as np
import tensorflow as tf


from Actor import Actor
from Critic import Critic
from ReplayBuffer import ReplayBuffer
from GameEngine import OpenAIGameEngine
# REPLAY BUFFER CONSTS
BUFFER_SIZE = 10000
BATCH_SIZE = 128
# FUTURE REWARD DECAY
GAMMA = 0.99

ENVIRONMENT_NAME = 'Pendulum-v0'
env = OpenAIGameEngine(ENVIRONMENT_NAME)



sess = tf.InteractiveSession()
state_size, action_size, action_high, action_low = env.gameParameters()
actor = Actor(sess, True, state_size, action_size, 200)
actor._session.run(tf.initialize_all_variables())
critic = Critic(sess, True, state_size, action_size, 300)
critic._session.run(tf.initialize_all_variables())
buffer = ReplayBuffer(BUFFER_SIZE)


def task():
    for ep in range(10000):
        s = env.initialState()
        Totoal = 0
        # what if the action is beyond the scope?
        for iteration  in range(100):
            # select the action with actor model.
            env.show()
            a = actor.getOutValueSmartly({"inputStates": [s]})[0] + (np.random.randn(1) / (ep + iteration + 1))
            r, s1, terminated = env.step(a)
            Totoal += r
            buffer.add(s, a, r, s1, terminated) #state, action, reward, new_state, done

            # update critic
            batch = buffer.getBatch(batch_size=BATCH_SIZE)
            S_ = [e[0] for e in batch]
            A_ = [e[1] for e in batch]
            R_ = [e[2] for e in batch]
            S1_ = [e[3] for e in batch]
            notTerminated = [1.-e[4] for e in batch]
            A1_ = actor.getOutValueSmartly({"inputStates":S1_})
            Q_S1_A1_ = critic.getOutValueSmartly({"inputStates":S1_, "inputActions":A1_})
            Ys = R_ + GAMMA * Q_S1_A1_ * notTerminated
            critic.minimizeSmartly({"inputStates":S_, "inputActions":A_, "inputYs":Ys})
            # update actor
            A = actor.getOutValueSmartly({"inputStates":S_})
            Qgradients = critic.anyNamedOperation("goa", {"inputStates":S_, "inputActions":A_}, False)[0]
            actor.ascentSmartly({"inputStates":S_, "Qgradients":Qgradients})

            s = s1

        print "EPISODE ", ep, "ENDED UP WITH REWARD: ", Totoal

# monitored run
env.monitor(task)
