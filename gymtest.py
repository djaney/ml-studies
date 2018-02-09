import random
import gym
import numpy as np
import tensorflow as tf
import numpy
from keras.models import Model
from keras.layers import Input, Dense
class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, strain_count = 10, mutation_chance=0.01):
        self.strains = []
        self.nextGen = []
        self.strain_count = strain_count
        self.model = self.createModel()
        self.mutation_chance = mutation_chance

        for _ in range(self.strain_count):
            self.strains.append(self.createModel().get_weights())

    def generationSize(self):
        return len(self.strains)

    def act(self, observation, strainIndex):
        self.model.set_weights(self.strains[strainIndex])
        y = self.model.predict(np.array([observation]), batch_size=1)
        res = numpy.argmax(y[0])
        return res,self.strains[strainIndex]

    def next(self, reward, strain):
        self.nextGen.append((reward,strain[0].flatten()))

    def createModel(self):
        a = Input(shape=(4,))
        b = Dense(2)(a)
        return Model(inputs=a, outputs=b)

    def evolve(self):

        # find fittest
        sorted(self.nextGen, key=lambda item: item[0], reverse=True)
        # breed
        father = self.nextGen.pop(0)[1]
        mother = self.nextGen.pop(0)[1]

        self.strains = []
        self.nextGen = []

        for _ in range(self.strain_count):
            newStrain = []
            for i in range(len(father)):
                if 0 == random.randrange(0,1):
                    newStrain.append(father[i])
                else:
                    newStrain.append(mother[i])

                # a chance to mutate
                if random.random() < self.mutation_chance:
                    mIdx = random.randrange(0, len(newStrain))
                    newStrain[mIdx] = random.uniform(-1, 1)+0.000001


            newStrain = numpy.reshape(newStrain, (4,2))
            self.strains.append([newStrain])




    
env = gym.make('CartPole-v0')
agent = Agent()

reward = 0
done = False
generationSize = 10
iterations = 100

# learn
gen = 0
while True:
    for strainIndex in range(agent.generationSize()):
        ob = env.reset()
        for t in range(1000):
            action = agent.act(ob, strainIndex)
            ob, reward, done, info = env.step(action[0])
            if done:
                agent.next( reward, action[1])
                break
    agent.evolve()
    if 0 == gen % 1000:
        print(gen, reward)
    gen=gen+1
