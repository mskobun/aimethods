import numpy as np
import pandas as pd
from genetic import GA
from pso import PSO


class GABacktest:
    def __init__(
        self,
        eps=0.01,
        delta=0.99,
        cardinality=10,
        pop_size=100,
        num_gens=200,
        trial_count=5,
    ):
        self.pop_size = pop_size
        self.num_gens = num_gens
        self.trial_count = trial_count
        self.eps = eps
        self.delta = delta
        self.cardinality = cardinality
        self.eps_floor = np.array([eps] * 101)
        self.delta_ceil = np.array([delta] * 101)

    def train(self):
        pass

    def get_weights(self, returns_arr: np.ndarray):
        df = pd.DataFrame(returns_arr)
        Genetic_Algorithm = GA(
            df,
            eps=self.eps_floor,
            delta=self.delta_ceil,
            cardinality=self.cardinality,
        )

        # Sample parameters
        pop_size = 500
        num_gens = 500
        trial_count = 5

        crsvr = "uniform_crossover"

        global_best_score = 0
        best_wts = []

        for i in range(0, trial_count):
            scores, wts, times, runtime = Genetic_Algorithm.solveGA(
                pop_size=pop_size,
                tournament_size=7,
                crossover_str=crsvr,
                crossover_rate=0.9,
                mutation_rate=0.2,
                num_generations=num_gens,
                elite_pct=0.1,
                patience_pct=0.2,
                seed=i + 1,
            )
            best_wts.append(wts)
            # scores : record of scores over the generations
            # wts : the record of best wts
            best_solution = Genetic_Algorithm.optimal_wts
            best_score = Genetic_Algorithm.sharpe(best_solution)
            if best_score > global_best_score:
                global_best_solution = best_solution
                global_best_score = best_score
        return global_best_solution


class PSOBacktest:
    def __init__(self, risk_free_rate: float = 0.0524 / 365, num_iterations: int = 500):
        self.risk_free_rate = risk_free_rate
        self.num_iterations = num_iterations

    def train(self):
        pass

    def get_weights(self, returns: np.ndarray):
        returns_df = pd.DataFrame(returns)
        pso = PSO(returns_df, risk_free_rate=self.risk_free_rate, quiet=True)
        weights, _, _ = pso.run(self.num_iterations)
        return weights
