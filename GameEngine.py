import gym
import cv2

class GameEngine:
    def __init__(self):
        print("game engine initialized")
    def initialState(self):
        raise NotImplementedError( "Should have implemented this, return R, S, T" )
    def step(self, action, state=None):
        raise NotImplementedError( "Should have implemented this" )

class OpenAIGameEngine(GameEngine):
    def __init__(self, ENVIRONMENT_NAME):
        self.envName = ENVIRONMENT_NAME
        self.env = gym.make(ENVIRONMENT_NAME)

    def initialState(self):
        s = self.env.reset()
        return s
    def step(self, action, state=None): # return r, s ,t
        s1, r, terminated, _ = self.env.step(action)
        return r, s1, terminated

    def gameParameters(self):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low
        return state_size, action_size, action_high, action_low
    def show(self):
        self.env.render()

    def monitor(self, task):
        self.env.monitor.start('experiments/' + self.envName,force=True)
        task()
        self.env.monitor.close()



class FlappyBirdGameEngine(GameEngine):
    def __init__(self):
        pass

    def initialState(self):
        import FlappyBirdEnv.wrapped_flappy_bird as game
        game_state = game.GameState()
        return game_state
    def step(self, action, state=None): # return r, s ,t
        ORIGIN_LENG = 160
        ORIGIN_WIDTH = 160
        x_t1_colored, r_t, terminal = state.frame_step(action)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (ORIGIN_LENG, ORIGIN_WIDTH)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        return ret, x_t1, terminal




