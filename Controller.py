import GameWorld
import matplotlib.pyplot as plt
import Agent

speed = 1
exampleGeneration = Agent.Generation(150)

for i in range(10):
    GameWorld.run_generation(exampleGeneration, speed, True)
    exampleGeneration.evolve(0.10)
