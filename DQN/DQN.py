#!/usr/bin/env python
from __future__ import print_function
import sys
import os

currentDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.abspath(os.path.join(currentDir,os.pardir)) # this will return parent directory.
sys.path.insert(0,parentDir)

from CNN import CNN
import tensorflow as tf
import cv2
from GameEngine import FlappyBirdGameEngine
import random
import numpy as np
from collections import deque

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

class QNetwork(CNN):
    def __init__(self, session, hasShadowNet): # create: G, sess, saver, savedDir. load: existing model if any
        CNN.__init__(self, session, hasShadowNet)
        self._screenInputs = self.buildInputLayer("_screenInputs", shape=[None, ORIGIN_LENG, ORIGIN_WIDTH, 4])
        tmp0 = self.buildConvReluWire(self._screenInputs, [4, 4, 4, 32], [1,4,4,1]) # stride=[1,4,4,1], where 4,4 are important
        out1 = self.buildMaxpoolWire(tmp0)
        tmp2 = self.buildConvReluWire(out1, [2, 2, 32, 64], [1,2,2,1]) # stride=[1,4,4,1], where 4,4 are important
        out2 = self.buildMaxpoolWire(tmp2)
        linearOut = self.buildFlattenWire(out2, [-1, 1600])
        tmp = self.buildLinearReluWire(linearOut, [1600, 512])
        self._qValuesForActions = self.buildLinearWire(tmp, [512, ACTIONS])
        self.setOutLayer(self._qValuesForActions)

        # train:
        #self._actionChoiceInputs = tf.placeholder(tf.float32, [None, ACTIONS])
        self._actionChoiceInputs = self.buildInputLayer("_actionChoiceInputs", shape=[None, ACTIONS])
        #self._targetValueInputs = tf.placeholder(tf.float32, [None])
        self._targetValueInputs = self.buildInputLayer("_targetValueInputs", shape=[None])

        #tf.sub does not exist, simply use "-"
        vvv = tf.multiply(self._actionChoiceInputs, self._qValuesForActions)
        self._observedValues = tf.reduce_sum(vvv, reduction_indices = 1)
        self._cost = tf.reduce_mean(tf.square(self._targetValueInputs - self._observedValues), 0)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(self._cost)
        self.addMinimizeOperation(self._optimizer)
        # self._G = tf.Graph()
        # self._Sess = tf.Session(graph=self._G)
        # with self._Sess.as_default():
        #     with self._G.as_default():
        #         self.createQNetwork()
        #
        #         tf.global_variables_initializer().run()
        #         self._saver = tf.train.Saver(tf.global_variables())
        #         self._savedDir = SAVEDIR_PREFIX
        #         checkpoint = tf.train.get_checkpoint_state(self._savedDir)
        #         if checkpoint and checkpoint.model_checkpoint_path:
        #             self._saver.restore(self._Sess, checkpoint.model_checkpoint_path)



    def getRewardScreen(self, game_state, action):
        x_t1_colored, r_t, terminal = game_state.frame_step(action)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (ORIGIN_LENG, ORIGIN_WIDTH)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        return ret, x_t1, terminal




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
                            #qvaluesForNewState = self._qValuesForActions.eval(feed_dict={self._screenInputs: [blockNews[i]]})[0]
                            qvaluesForNewState = self.getOutValueSmartly({"_screenInputs": [blockNews[i]]})[0]
                            targetValueInputVals[i] = rewards[i] + GAMMA * np.max(qvaluesForNewState, axis=0)
                        # else: # from reference net
                        #     qvaluesForNewState = referenceNet._qValuesForActions.eval(feed_dict={referenceNet._screenInputs: [blockNews[i]]})[0]
                        #     targetValueInputVals[i] = rewards[i] + GAMMA * np.max(qvaluesForNewState, axis=0)


                self.minimizeSmartly({"_screenInputs": screenInputVals, "_actionChoiceInputs": actionChoiceInputVals, "_targetValueInputs":targetValueInputVals})
                #self._optimizer.run(feed_dict={self._screenInputs: screenInputVals, self._actionChoiceInputs: actionChoiceInputVals, self._targetValueInputs:targetValueInputVals})

            if qvaluesForActions == None:
                print("TIMESTEP", counter,  "/ ACTION", np.argmax(action), "/ REWARD", reward, "/ terminal ",  terminal)
            else:
                print("TIMESTEP", counter,  "/ ACTION", np.argmax(action), "/ REWARD", reward, "/ terminal ",  terminal, "/ Q_MAX ", np.max(qvaluesForActions))

            block = blockNew


            # if counter % SAVING_PERIOD == 0:
            #     if not os.path.exists(self._savedDir):
            #         os.makedirs(self._savedDir)
            #     self._saver.save(self._Sess, self._savedDir + '/' + MODELNAME) # overwrite the last iteration







# iteartively refine the representation of the images.
# the network trained in last iteration using the coarser images are used to guide the current iteartion



def main():
    sess = tf.InteractiveSession()
    qNet = QNetwork(sess, True)
    tf.global_variables_initializer().run()

    qNet.control()



if __name__ == "__main__":
    main()
