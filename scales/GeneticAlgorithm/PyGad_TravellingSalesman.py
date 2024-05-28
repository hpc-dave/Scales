import numpy as np
import random
import pygad

distance_matrix = np.array([[0, 10, 15, 20],
                            [10, 0, 35, 25],
                            [15, 35, 0, 30],
                            [20, 25, 30, 0]])


def fitness_func(ga, solution, solution_idx):
    route = np.array(solution)
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i], route[i+1]]
    distance += distance_matrix[route[-1], route[0]]
    fitness = 1.0 / distance
    return fitness


num_generations = 100
population_size = 50
num_parents_mating = 2
mutation_probability = 0.01
num_genes = len(distance_matrix)
initial_population = [[random.randint(0, num_genes-1) for _ in range(0, num_genes)] for _ in range(population_size)]
ga_instance = pygad.GA(initial_population=initial_population,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       mutation_probability=mutation_probability,
                       gene_type=int,
                       allow_duplicate_genes=False,
                       gene_space=[i for i in range(0, num_genes)])
ga_instance.run()

best_solution, best_solution_fitness, best_index = ga_instance.best_solution()

print("Initial solution:", initial_population)
print("Best solution:", best_solution)
print("Best solution fitness:", best_solution_fitness)

print('finished')