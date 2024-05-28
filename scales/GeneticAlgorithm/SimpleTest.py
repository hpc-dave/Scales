import pygad
import random
import math


def fitness_function(x):
    return 1-(x - 0.5)**2


def evaluate_population(population):
    return [fitness_function(individual) for individual in population]


def select_parents(population, scores):
    rank = [i for i in range(len(scores))]
    rank.sort(key=lambda x: scores[x], reverse=True)
    rank_prob = [(( 2 * len(scores) - i ) / (2 * len(scores))) for i in range(len(scores))]
    parent_indices = []
    for _ in range(2):
        while(len(parent_indices) < 2):
            random_number = random.uniform(0, 1)
            for i, p in enumerate(rank_prob):
                if random_number < p and rank[i] not in parent_indices:
                    parent_indices.append(rank[i])
                    break
    return [population[i] for i in parent_indices]


def crossover(parents):
    if isinstance(parents[0], float) or isinstance(parents[0], int):
        point = random.uniform(0, 1)
        child1 = parents[0] * point + parents[1] * (1. - point)
        child2 = parents[1] * point + parents[0] * (1. - point)
    else:
        point = random.randint(1, len(parents[0]) - 1)
        child1 = parents[0][:point] + parents[1][point:]
        child2 = parents[1][:point] + parents[0][point:]
    return [child1, child2]


def mutate(child, mutation_rate):
    if isinstance(child, float):
        if random.uniform(0, 1) < mutation_rate:
            child *= random.uniform(0, 1)
    else:
        for i in range(len(child)):
            if random.uniform(0, 1) < mutation_rate:
                child[i] = random.uniform(0, 1)
    return child


def genetic_algorithm(population_size, num_generations, mutation_rate):
    # Step 1: Initialization
    population = [random.uniform(0, 1) for _ in range(population_size)]
 
    for generation in range(num_generations):
        # Step 2: Evaluation
        scores = evaluate_population(population)
 
        # Step 3: Selection
        parents = [select_parents(population, scores) for _ in range(population_size // 2)]
 
        # Step 4: Crossover
        children = [crossover(p) for p in parents]
        children = [item for sublist in children for item in sublist]
 
        # Step 5: Mutation
        mutated_children = [mutate(c, mutation_rate) for c in children]
 
        # Combine parents and children
        parents = [item for sublist in parents for item in sublist]
        combined_population = parents + mutated_children
 
        # Step 2: Evaluation
        scores = evaluate_population(combined_population)
 
        # Select the fittest individuals
        ranked_population = [x for _, x in sorted(zip(scores, combined_population), reverse=True)]
        population = ranked_population[:population_size]
 
    return population[0]


res = genetic_algorithm(population_size=10, num_generations=1, mutation_rate=1.)

print('finished')