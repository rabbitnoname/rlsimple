#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
from GameEngine import FlappyBirdGameEngine
import random
import numpy as np
from collections import deque
import os

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.01 # starting value of epsilon lpxz
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
ORIGIN_LENG = 160
ORIGIN_WIDTH = 160
SAVEDIR_PREFIX = "saved_networks"
MODELNAME = "dqn"
SAVING_PERIOD = 10000

game = FlappyBirdGameEngine()

class QNetwork():
    def __init__(self, iterationID): # create: G, sess, saver, savedDir. load: existing model if any
        self._G = tf.Graph()
        self._Sess = tf.Session(graph=self._G)
        with self._Sess.as_default():
            with self._G.as_default():
                self.createQNetwork()

                tf.global_variables_initializer().run()
                self._saver = tf.train.Saver(tf.global_variables())
                self._savedDir = SAVEDIR_PREFIX + str(iterationID)
                checkpoint = tf.train.get_checkpoint_state(self._savedDir)
                if checkpoint and checkpoint.model_checkpoint_path:
                    self._saver.restore(self._Sess, checkpoint.model_checkpoint_path)


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def getRewardScreen(self, game_state, action):
        x_t1_colored, r_t, terminal = game_state.frame_step(action)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (ORIGIN_LENG, ORIGIN_WIDTH)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        return ret, x_t1, terminal


    def createQNetwork(self):
        self._screenInputs = tf.placeholder(tf.float32, [None, ORIGIN_LENG, ORIGIN_WIDTH, 4])
        W1 = tf.Variable(tf.truncated_normal([4, 4, 4, 32], stddev = 0.01)) # lpxz: different from the working version
        B1 = tf.Variable(tf.truncated_normal([32], stddev = 0.01))
        tmp0 = tf.nn.relu(tf.nn.bias_add(self.conv2d(self._screenInputs, W1, 4), B1))
        out1 = self.max_pool_2x2(tmp0)
        W2 = tf.Variable(tf.truncated_normal([2, 2, 32, 64], stddev = 0.01)) # lpxz: different from the working version
        B2 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
        tmp2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(out1, W2, 2), B2))
        out2 = self.max_pool_2x2(tmp2)
        linearOut = tf.reshape(out2, [-1, 1600])
        W3 = tf.Variable(tf.truncated_normal([1600, 512], stddev = 0.01))
        B3 = tf.Variable(tf.truncated_normal([512], stddev = 0.01))
        tmp = tf.nn.relu(tf.add(tf.matmul(linearOut, W3), B3))
        W4 = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev = 0.01))
        B4 = tf.Variable(tf.truncated_normal([ACTIONS], stddev = 0.01))
        self._qValuesForActions = tf.add(tf.matmul(tmp, W4), B4)

        # train:
        self._actionChoiceInputs = tf.placeholder(tf.float32, [None, ACTIONS])
        self._targetValueInputs = tf.placeholder(tf.float32, [None])

        #tf.sub does not exist, simply use "-"
        vvv = tf.multiply(self._actionChoiceInputs, self._qValuesForActions)
        self._observedValues = tf.reduce_sum(vvv, reduction_indices = 1)
        self._cost = tf.reduce_mean(tf.square(self._targetValueInputs - self._observedValues), 0)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(self._cost)


    def makeImageBlock(self,image):
        return np.stack((image, image, image, image), axis=2)

    def updateImageBlock(self,block, image):
        return np.append(block[:, :, 1:], image, axis=2)

    def control(self, referenceNet=None):
        action = np.zeros([ACTIONS])
        action[0] = 1
        game_state = game.initialState()
        reward, x_t1, _ = game.step(action, game_state)
        block = self.makeImageBlock(x_t1)

        replayBuffer = deque()
        episilon = INITIAL_EPSILON
        counter = 0
        while True:
            counter += 1
            action = np.zeros([ACTIONS])

            # read -> episilon greedy control
            if random.random() < episilon:
                randIndex = random.randint(0, ACTIONS-1)
                action[randIndex] = 1
                qvaluesForActions = None # do not inherit from last timestamp
            else:
                with self._Sess.as_default():
                    with self._G.as_default():
                        qvaluesForActions = self._qValuesForActions.eval(feed_dict={self._screenInputs: [block]})[0]
                        index = np.argmax(qvaluesForActions)
                        action[index] = 1

            # scale down epsilon
            if episilon > FINAL_EPSILON and counter > OBSERVE:
                episilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            reward, x_t1, terminal = game.step(action, game_state)

            x_t1 = np.reshape(x_t1, (ORIGIN_LENG, ORIGIN_WIDTH, 1))
            blockNew = self.updateImageBlock(block, x_t1)

            # feedback -> replay buffer
            if len(replayBuffer) >= REPLAY_MEMORY:
                replayBuffer.popleft()

            replayBuffer.append((block, action, reward, terminal, blockNew))
            if counter >= OBSERVE:
                # train
                batchObservations = random.sample(replayBuffer, BATCH)
                screenInputVals = [x[0] for x in batchObservations]
                actionChoiceInputVals = [x[1] for x in batchObservations]
                rewards = [x[2] for x in batchObservations]
                terminals = [x[3] for x in batchObservations]
                blockNews = [x[4] for x in batchObservations]
                targetValueInputVals = np.zeros([BATCH])
                for i in range(BATCH):
                    if terminals[i]:
                        targetValueInputVals[i] = rewards[i]
                    else:
                        # target value.
                        if referenceNet == None: # from self
                            with self._Sess.as_default():
                                with self._G.as_default():
                                    qvaluesForNewState = self._qValuesForActions.eval(feed_dict={self._screenInputs: [blockNews[i]]})[0]
                                    targetValueInputVals[i] = rewards[i] + GAMMA * np.max(qvaluesForNewState, axis=0)
                        else: # from reference net
                            with referenceNet._Sess.as_default():
                                with referenceNet._G.as_default():
                                    qvaluesForNewState = referenceNet._qValuesForActions.eval(feed_dict={referenceNet._screenInputs: [blockNews[i]]})[0]
                                    targetValueInputVals[i] = rewards[i] + GAMMA * np.max(qvaluesForNewState, axis=0)


                with self._Sess.as_default():
                    with self._G.as_default():
                        self._optimizer.run(feed_dict={self._screenInputs: screenInputVals, self._actionChoiceInputs: actionChoiceInputVals, self._targetValueInputs:targetValueInputVals})

            if qvaluesForActions == None:
                print("TIMESTEP", counter,  "/ ACTION", np.argmax(action), "/ REWARD", reward, "/ terminal ",  terminal)
            else:
                print("TIMESTEP", counter,  "/ ACTION", np.argmax(action), "/ REWARD", reward, "/ terminal ",  terminal, "/ Q_MAX ", np.max(qvaluesForActions))

            block = blockNew


            if counter % SAVING_PERIOD == 0:
                if not os.path.exists(self._savedDir):
                    os.makedirs(self._savedDir)
                self._saver.save(self._Sess, self._savedDir + '/' + MODELNAME) # overwrite the last iteration







# iteartively refine the representation of the images.
# the network trained in last iteration using the coarser images are used to guide the current iteartion



def main():
    qNetOld = QNetwork(iterationID=0)
    qNet = QNetwork(iterationID=1)
    qNet.control(referenceNet=qNetOld)



if __name__ == "__main__":
    main()
