import numpy as np 
import matplotlib.pyplot as plt
import random 

# Class to represent city name and coordinates
class City:
    def __init__(self,x,y,name):
        self.x = x
        self.y = y
        self.name = name
    
    def distance_to(self,city):
        return np.sqrt((self.x - city.x)**2 + (self.y - city.y)**2)

# Class to store and calculate fitness of a route
class Fitness:
    def __init__(self,route):
        self.route = route
        self.distance = self.calculate_total_distance()
        self.fitness = 1/self.distance

    def calculate_total_distance(self):
        distance = 0
        for i in range(len(self.route)):
            start_city = route[i]
            next_city = route[(i+1) % len(self.route)]
            # wrap around to first city
            distance += start_city.distance_to(next_city)                   
        return distance

# Main class implementing Genetic Algorithm by TSP
class GeneticAlgorithmTSP:
    def __init__(self,cities,population_size = 100):
        self.cities = cities
        self.population_size = population_size
        self.population = []
        self.fitness_result = []


    # create a rondom route by shuffling the cities
    def create_route(self):
        return np.sample(self.cities,len(self.cities))
    
    # initialize population with random routes
    def initial_population(self):
        self.population = [self.create_route() for _ in range(self.population_size)]
    
    def rank_population(self):
        self.fitness_result = [Fitness(route) for route in self.population]
        # sort routes in descending by fitness
        self.fitness_result.sort(key=lambda x:x[1].fitness,reverse=True)        
    
    def crossover(self,parent1,parent2):

    


        