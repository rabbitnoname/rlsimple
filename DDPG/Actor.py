import tensorflow as tf
from NN import NN
class Actor(NN):
    # variables into a bag
    # ema wrapper
    # gradient wrapper
    def __init__(self, session, hasShadowNet, state_size, action_size, hidden_state_size):
        NN.__init__(self, session, hasShadowNet)
        # nothing special:
        inputLayer = self.buildInputLayer("inputStates", shape=[None, state_size])
        h1 = self.buildLinearReluWire(inputLayer, [state_size, hidden_state_size])
        #for i in range(numOfHiddenLayers-1): # repeat (numOfHiddenLayers-1) times
        h1 = self.buildLinearReluWire(h1, [hidden_state_size, hidden_state_size])
        out = self.buildLinearWire(h1, [hidden_state_size, action_size])
        self.setOutLayer(out)

        # a bit unique:
        Qgradient = self.buildInputLayer("Qgradients", shape=[None, action_size])
        self.addAscentOperation(Qgradient)
