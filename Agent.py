import NeuralNetwork
import random

# Constants
JUMP_FORCE = 4
NETWORK_TOPOLOGY = [3, 3, 3, 1]

class AgentEntity:
    """
    Represents an agent in the Flappy Bird game.
    """
    def __init__(self, neural_network=None):
        """
        Initialize an Agent instance.

        :param neural_network: Optional neural network for the agent.
        :type neural_network: NeuralNetwork.NeuralNetwork
        """
        self.agent_velocity = 0  # velocity of the agent
        self.agent_position = [0, 0]
        self.radius = 10  # Size of the agent
        self.score = 0  # Current score of the agent
        self.goals = 0
        self.active = True  # Flag to determine if the agent is active or not
        self.neural_network = neural_network if neural_network else NeuralNetwork.NeuralNetwork(NETWORK_TOPOLOGY)  # Example architecture for the neural network

    def jump(self):
        """Make the agent jump by applying an upward force."""
        self.agent_velocity = -JUMP_FORCE

    def move(self):
        """
        Move the agent based on inputs and gravity.

        :param inputs: List of inputs for the neural network.
        :type inputs: list
        :param gravity: Gravity affecting the agent's vertical motion.
        :type gravity: float
        """
        self.agent_position[1] += self.agent_velocity

    def process(self, inputs, gravity):
        self.agent_velocity += gravity

        prediction = self.neural_network.predict(inputs)
        if prediction[0] > 0.5:
            self.jump()

    def reset(self, position, velocity):
        """
        Reset the agent's state.

        :param position: Initial vertical position of the agent.
        :type position: int
        """
        self.agent_velocity = velocity
        self.agent_position = [position[0], position[1]]
        self.active = True  # Reactivate the agent
        self.score = 0
        self.goals = 0

    def copy(self):
        """
        Create a copy of the agent with the same neural network architecture.

        :return: Copied agent with identical neural network.
        :rtype: Agent
        """
        return AgentEntity(neural_network=self.neural_network.copy())

    def reproduce(self):
        """
        Create offspring by mutating the agent's neural network.

        :return: Offspring agent with a mutated neural network.
        :rtype: Agent
        """
        child_neural_network = self.neural_network.copy()
        child_neural_network.mutate()

        return AgentEntity(neural_network=child_neural_network)

class Generation:
    """
    Represents a generation of agents in the Flappy Bird game.
    """
    def __init__(self, population_size):
        """
        Initialize a Generation instance with a specified population size.

        :param population_size: Size of the population (number of agents).
        :type population_size: int
        """
        self.population_size = population_size
        self.population = list()
        self.population = [AgentEntity() for _ in range(population_size)]  # Create a population of agents
        self.number = 0

    def reset_population(self, position, velocity):
        """
        Reset the entire population of agents to a specific initial position.

        :param position: Initial vertical position for all agents in the population.
        :type position: int
        """
        for i in range(len(self.population)):
            self.population[i].reset(position, velocity)

    def evolve(self, threshold):
        """
        Evolve the population by selecting top-performing agents as survivors and creating offspring.

        :param threshold: Fraction of top-performing agents to keep as survivors.
        :type threshold: float
        """
        self.number += 1

        self.population.sort(key=lambda bird: bird.score, reverse=True)  # Sort agents by their scores in descending order
        survivors_count = int(self.population_size * threshold)
        survivors = self.population[:survivors_count]  # Select the top-performing agents as survivors

        self.population = list()
        self.population.extend(survivors)  # Replace the population with survivors

        while len(self.population) < self.population_size:
            parent = random.choice(survivors)  # Randomly select a parent from survivors
            child = parent.reproduce()  # Create a child agent by mutating the parent's neural network
            self.population.append(child)  # Add the child to the new population

    def get_avg_score(self):
        return sum(agent.score for agent in self.population) / len(self.population)
