import gym

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

env = gym.make('CartPole-v0')
agent = Agent(env.action_space)

reward = 0
done = False

# learn
for e in range(1000):
    ob = env.reset()
    for t in range(100):
        action = agent.act(ob)
        ob, reward, done, info = env.step(action)
        if done:
            # learing algorithm here
            break


# test
ob = env.reset()
reward = 0
done = False

for t in range(100):
    env.render()
    action = agent.act(ob)
    ob, reward, done, info = env.step(action)
    if done: break
