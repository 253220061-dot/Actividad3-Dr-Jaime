"""
ACTIVIDAD 03 - ALGORITMOS EVOLUTIVOS (UPMH)
Programa: Algoritmo Genético de Codificación Binaria para Selección de Cartera de Proyectos con Sinergias
Autor: Eduardo Taurino Martínez Morales
Docente: Dr. Jaime Aguilar Ortiz
"""

import numpy as np
import pandas as pd

# Fijar semilla para reproducibilidad
np.random.seed(42)

# 1. PARÁMETROS DEL PROBLEMA (n = 15 proyectos)
n = 15
npv = np.array([45, 30, 60, 25, 50, 40, 35, 70, 20, 55, 65, 80, 28, 48, 52])  # VPN (MUSD)
cost = np.array([12, 8, 18, 6, 15, 10, 9, 22, 5, 16, 20, 25, 7, 14, 15])     # Costo (MUSD)
personnel = np.array([5, 3, 7, 2, 6, 4, 3, 8, 2, 6, 7, 9, 3, 5, 6])          # Personal
risk = np.array([3, 2, 5, 1, 4, 3, 2, 6, 1, 4, 5, 7, 2, 4, 4])               # Riesgo

# Matriz de Sinergias S (15x15)
S = np.zeros((n, n))
# Sinergias positivas (+)
S[0, 1] = S[1, 0] = 8.0
S[2, 4] = S[4, 2] = 12.0
S[7, 10] = S[10, 7] = 15.0
S[3, 8] = S[8, 3] = 6.0
S[11, 13] = S[13, 11] = 10.0
S[5, 14] = S[14, 5] = 9.0

# Interferencias negativas (-)
S[0, 2] = S[2, 0] = -10.0
S[6, 7] = S[7, 6] = -8.0
S[9, 11] = S[11, 9] = -12.0

# Techos del sistema
C_max = 90.0      # Presupuesto Financiero Máximo
R_max = 40        # Personal Máximo
W_max = 30        # Riesgo Máximo
rho = 10.0        # Penalizador por unidad de exceso

# 2. FUNCIÓN DE EVALUACIÓN
def evaluate_individual(x):
    x = np.asarray(x, dtype=int)
    ind_npv = np.sum(npv * x)
    synergy = 0.5 * np.dot(x, np.dot(S, x))
    total_benefit = ind_npv + synergy
    
    v_cost = max(0.0, np.sum(cost * x) - C_max)
    v_pers = max(0.0, np.sum(personnel * x) - R_max)
    v_risk = max(0.0, np.sum(risk * x) - W_max)
    
    total_viol = v_cost + v_pers + v_risk
    fitness_J = total_benefit - rho * total_viol
    
    return {
        'x': x,
        'benefit': total_benefit,
        'ind_npv': ind_npv,
        'synergy': synergy,
        'cost': np.sum(cost * x),
        'personnel': np.sum(personnel * x),
        'risk': np.sum(risk * x),
        'total_viol': total_viol,
        'J': fitness_J,
        'feasible': (total_viol == 0)
    }

def hamming_diversity(pop):
    N, n_genes = pop.shape
    if N <= 1: return 0.0
    dist_sum = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist_sum += np.sum(pop[i] != pop[j])
            count += 1
    return dist_sum / (count * n_genes)

def linear_ranking_selection(pop, fitnesses, s=1.7):
    N = len(pop)
    ranks = np.argsort(np.argsort(fitnesses))
    probs = (2.0 - s) / N + (2.0 * ranks * (s - 1.0)) / (N * (N - 1.0))
    probs = np.maximum(0, probs)
    probs /= np.sum(probs)
    selected_indices = np.random.choice(N, size=N, p=probs)
    return pop[selected_indices]

def uniform_crossover(p1, p2, pc=0.85):
    if np.random.rand() > pc:
        return p1.copy(), p2.copy()
    mask = np.random.rand(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2

def bitflip_mutation(ind, pm=1.0/n):
    mutated = ind.copy()
    for i in range(len(ind)):
        if np.random.rand() < pm:
            mutated[i] = 1 - mutated[i]
    return mutated

# 3. EJECUCIÓN PRINCIPAL DEL ALGORITMO GENÉTICO
def run_ga(pop_size=30, max_generations=50, seed=42):
    np.random.seed(seed)
    pop = np.zeros((pop_size, n), dtype=int)
    n_greedy = int(0.15 * pop_size)
    
    density = npv / cost
    greedy_order = np.argsort(-density)
    
    for i in range(n_greedy):
        ind = np.zeros(n, dtype=int)
        curr_c = 0
        for idx in greedy_order:
            if curr_c + cost[idx] <= C_max and np.random.rand() > 0.1:
                ind[idx] = 1
                curr_c += cost[idx]
        pop[i] = ind
        
    for i in range(n_greedy, pop_size):
        pop[i] = np.random.binomial(1, 0.4, size=n)
        
    best_global_J = -1e9
    best_global_ind = None
    eval_count = 0
    stagnation = 0
    
    for gen in range(max_generations):
        evals = [evaluate_individual(ind) for ind in pop]
        eval_count += pop_size
        fitnesses = np.array([e['J'] for e in evals])
        
        gen_best_idx = np.argmax(fitnesses)
        gen_best_e = evals[gen_best_idx]
        
        if gen_best_e['J'] > best_global_J:
            best_global_J = gen_best_e['J']
            best_global_ind = gen_best_e
            stagnation = 0
        else:
            stagnation += 1
            
        div = hamming_diversity(pop)
        if eval_count >= 1500 or stagnation >= 15 or div <= 0.01:
            break
            
        mating_pool = linear_ranking_selection(pop, fitnesses, s=1.7)
        next_pop = []
        for i in range(0, pop_size, 2):
            p1, p2 = mating_pool[i], mating_pool[(i+1)%pop_size]
            c1, c2 = uniform_crossover(p1, p2, pc=0.85)
            c1 = bitflip_mutation(c1, pm=1.0/n)
            c2 = bitflip_mutation(c2, pm=1.0/n)
            next_pop.extend([c1, c2])
            
        next_pop = np.array(next_pop[:pop_size])
        elite_indices = np.argsort(fitnesses)[-2:]
        next_pop[0] = pop[elite_indices[1]].copy()
        next_pop[1] = pop[elite_indices[0]].copy()
        pop = next_pop
        
    return best_global_ind, eval_count

if __name__ == "__main__":
    best_solution, total_evals = run_ga()
    print("=== RESULTADO ÓPTIMO ALCANZADO POR EL ALGORITMO GENÉTICO ===")
    print("Vector de Cartera Óptima (x*):", best_solution['x'])
    print("Valor de Aptitud J(x*):", best_solution['J'])
    print("Beneficio Total B(x*):", best_solution['benefit'], "MUSD")
    print("  - VPN Individual Base:", best_solution['ind_npv'], "MUSD")
    print("  - Valor Agregado por Sinergias:", best_solution['synergy'], "MUSD")
    print("Consumo de Recursos:")
    print("  - Presupuesto Financiero:", best_solution['cost'], "/ 90.0 MUSD")
    print("  - Personal Técnico:", best_solution['personnel'], "/ 40 Especialistas")
    print("  - Índice de Riesgo:", best_solution['risk'], "/ 30 Puntos")
    print("Factibilidad:", best_solution['feasible'])
    print("Evaluaciones Consumidas:", total_evals)