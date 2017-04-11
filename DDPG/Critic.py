import tensorflow as tf
from NN import NN
class Critic(NN):
    def __init__(self, session, hasShadowNet, state_size, action_size, hidden_state_size):
        NN.__init__(self, session, hasShadowNet)
        # nothing special:
        inputStates = self.buildInputLayer("inputStates", shape=[None, state_size])
        inputActions = self.buildInputLayer("inputActions", shape=[None, action_size])
        inputYs = self.buildInputLayer("inputYs", shape=[None])
        h1 = self.buildLinearReluWire(inputStates, [state_size, hidden_state_size])
        #for i in range(numOfHiddenLayers-1): # repeat (numOfHiddenLayers-1) times
        h1 = self.buildLinearReluWire(h1, [hidden_state_size, hidden_state_size])
        h1 = self.buildJointLinearReluWire(h1, [hidden_state_size, hidden_state_size], inputActions, [action_size, hidden_state_size])
        tmp1 = self.buildLinearWire(h1, [hidden_state_size, action_size])
        out = self.buildReduceSum(tmp1, reduction_indices=1)
        self.setOutLayer(out)

        # a bit unique:
        # action_size =1, so we do not need reduce_sum or action_choice
        self.error = tf.reduce_mean(tf.square(inputYs - out)) # all, every row has only one, let us take fast path
        self.addMinimizeOperation(tf.train.AdamOptimizer(0.001).minimize(self.error))
        self.addAnyNamedOperation("goa", tf.gradients(out, inputActions))


