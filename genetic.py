# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json


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
        returns_df = pd.DataFrame(returns_arr)

        self.genetic_algorithm = GA(
            returns_df,
            eps=self.eps_floor,
            delta=self.delta_ceil,
            cardinality=self.cardinality,
        )
        selection_strs = ["Tournament"]
        crsvr_strs = ["uniform_crossover"]

        runtime_traces = {
            crsvr: {
                select: {trial: np.array([]) for trial in range(self.trial_count)}
                for select in selection_strs
            }
            for crsvr in crsvr_strs
        }
        fitness_traces = {
            crsvr: {
                select: {trial: np.array([]) for trial in range(self.trial_count)}
                for select in selection_strs
            }
            for crsvr in crsvr_strs
        }
        final_scores = {
            crsvr: {
                select: {trial: np.array([]) for trial in range(self.trial_count)}
                for select in selection_strs
            }
            for crsvr in crsvr_strs
        }
        score_plot_region = {
            crsvr_strs[i - 1]: 200 + len(crsvr_strs) * 10 + i
            for i in range(1, len(crsvr_strs) + 1)
        }
        runtime_plot_region = {
            crsvr_strs[i - 1]: 200 + len(crsvr_strs) * 10 + i + len(crsvr_strs)
            for i in range(1, len(crsvr_strs) + 1)
        }

        global_best_score = 0
        best_wts = []
        for crsvr in crsvr_strs:
            for selection in selection_strs:
                for i in range(0, self.trial_count):
                    scores, wts, times, runtime = self.genetic_algorithm.solveGA(
                        pop_size=self.pop_size,
                        tournament_size=5,
                        crossover_str=crsvr,
                        selection_str=selection,
                        crossover_rate=0.85,
                        mutation_rate=0.0,
                        num_generations=self.num_gens,
                        elite_pct=0.1,
                        patience_pct=0.2,
                        seed=i + 1,
                    )
                    best_wts.append(wts)
                    # scores : record of scores over the generations
                    # wts : the record of wts
                    best_solution = self.genetic_algorithm.optimal_wts
                    best_score = self.genetic_algorithm.sharpe(best_solution)
                    if best_score > global_best_score:
                        global_best_solution = best_solution
                        global_best_score = best_score
                        global_best_trial = i + 1
                        global_best_time = runtime

                    trial_opt_weights = self.genetic_algorithm.optimal_wts
                    final_scores[crsvr][selection][i] = self.genetic_algorithm.sharpe(
                        trial_opt_weights
                    )
                    runtime_traces[crsvr][selection][i] = np.array(times)
                    fitness_traces[crsvr][selection][i] = scores

                solution = self.genetic_algorithm.optimal_wts
                return solution


# %%
def main():
    eps_floor = np.array([0.01] * 101)
    delta_ceil = np.array([0.99] * 101)
    data_wd = "data/return_df.csv"
    returns_df = pd.read_csv(data_wd, index_col="Date")
    card = 10
    Genetic_Algorithm = GA(
        returns_df, eps=eps_floor, delta=delta_ceil, cardinality=card
    )
    fig = plt.figure(figsize=(36, 12))

    pop_size = 100
    num_gens = 200
    trial_count = 5

    linestyles = {"Roulette": "-", "Tournament": "--"}

    selection_strs = ["Tournament"]
    crsvr_strs = ["uniform_crossover"]

    runtime_traces = {
        crsvr: {
            select: {trial: np.array([]) for trial in range(trial_count)}
            for select in selection_strs
        }
        for crsvr in crsvr_strs
    }
    fitness_traces = {
        crsvr: {
            select: {trial: np.array([]) for trial in range(trial_count)}
            for select in selection_strs
        }
        for crsvr in crsvr_strs
    }
    final_scores = {
        crsvr: {
            select: {trial: np.array([]) for trial in range(trial_count)}
            for select in selection_strs
        }
        for crsvr in crsvr_strs
    }
    score_plot_region = {
        crsvr_strs[i - 1]: 200 + len(crsvr_strs) * 10 + i
        for i in range(1, len(crsvr_strs) + 1)
    }
    runtime_plot_region = {
        crsvr_strs[i - 1]: 200 + len(crsvr_strs) * 10 + i + len(crsvr_strs)
        for i in range(1, len(crsvr_strs) + 1)
    }

    fig = plt.figure(figsize=(32, 12))
    global_best_score = 0
    best_wts = []
    for crsvr in crsvr_strs:
        ax_scores = fig.add_subplot(score_plot_region[crsvr])
        ax_runtime = fig.add_subplot(runtime_plot_region[crsvr])
        for selection in selection_strs:
            print(f"Running Genetic Algorithm ({crsvr}, {selection}) ...")
            print(f"eps = {eps_floor}")
            print(f"delta = {delta_ceil}")
            print(
                f"{'Trial':<6} {'Seed':<6} {'Best_Score':<12} {'Time Taken (s)':<14} {'Number of Generations':<100}"
            )
            print("-" * 60)
            for i in range(0, trial_count):
                scores, wts, times, runtime = Genetic_Algorithm.solveGA(
                    pop_size=pop_size,
                    tournament_size=5,
                    crossover_str=crsvr,
                    selection_str=selection,
                    crossover_rate=0.85,
                    mutation_rate=0.0,
                    num_generations=num_gens,
                    elite_pct=0.1,
                    patience_pct=0.2,
                    seed=i + 1,
                )
                best_wts.append(wts)
                # scores : record of scores over the generations
                # wts : the record of wts
                best_solution = Genetic_Algorithm.optimal_wts
                best_score = Genetic_Algorithm.sharpe(best_solution)
                if best_score > global_best_score:
                    global_best_solution = best_solution
                    global_best_score = best_score
                    global_best_trial = i + 1
                    global_best_time = runtime

                trial_opt_weights = Genetic_Algorithm.optimal_wts
                final_scores[crsvr][selection][i] = Genetic_Algorithm.sharpe(
                    trial_opt_weights
                )
                runtime_traces[crsvr][selection][i] = np.array(times)
                fitness_traces[crsvr][selection][i] = scores
                ax_scores.plot(
                    range(len(scores)),
                    scores,
                    label=f"Trial {i + 1} ({selection})",
                    linestyle=linestyles[selection],
                )
                ax_runtime.plot(
                    range(len(times)),
                    times,
                    label=f"Trial {i + 1} ({selection})",
                    linestyle=linestyles[selection],
                )
                print(
                    f"{i + 1:<6} {i + 1:<6} {best_score:<12.6f} {runtime:<14.6f} {len(scores):<21} \n{Genetic_Algorithm.optimal_wts} {np.sum(Genetic_Algorithm.optimal_wts)}"
                )

            print("\n\n")
            ax_scores.set_title(f"Fitness Traces across all trials {crsvr}")
            ax_scores.set_xlabel("Generation")

            ax_scores.set_ylabel("Objective")

            ax_runtime.set_title(f"Runtime Traces across all trials {crsvr}")
            ax_runtime.set_xlabel("Generation")
            ax_runtime.set_ylabel("Time (s)")

            ax_scores.grid(True)
            ax_scores.legend(loc="lower right")

            ax_runtime.grid(True)
            ax_runtime.legend(loc="lower right")
            daily_returns = Genetic_Algorithm.returns
            solution = Genetic_Algorithm.optimal_wts
            print(np.sum(Genetic_Algorithm.optimal_wts != 0))
            portfolio = dict(zip(daily_returns, solution))
            holdings = {k: v for k, v in portfolio.items() if v > 0}
            sorted_holdings = dict(
                sorted(holdings.items(), key=lambda x: x[1], reverse=True)
            )
            portfolio_returns = daily_returns @ solution

            portfolio_cum_return = (1 + portfolio_returns).prod() - 1

            print("Summary of GA:")
            print("\nOptimal Portfolio Composition:")
            print("------------------------------")
            for stock, weight in sorted_holdings.items():
                print(f"{stock}: {weight:.4f} ({weight * 100:.2f}%)")
            print(f"\nPerformance Comparison:")
            print(f"Portfolio Sharpe: {Genetic_Algorithm.sharpe(solution):.4f}")
            print(f"Portfolio Cumulative Return: {portfolio_cum_return:.2%}")

            with open("GA_holdings.json", "w") as file:
                json.dump(sorted_holdings, file, indent=4)


class snp100_Portfolio:
    def __init__(
        self,
        returns_df: pd.DataFrame,
    ):
        # Return Data
        self.returns = returns_df
        self.mean_returns = np.array(
            [np.mean(self.returns[a]) for a in self.returns.columns]
        )
        self.cov_matrix = self.returns.cov().values
        self.assets = self.returns.columns


class GA(snp100_Portfolio):
    def __init__(
        self,
        returns_df: pd.DataFrame,
        eps=np.array([0.1] * 101),
        delta=np.array([0.9] * 101),
        cardinality=20,
    ):
        super().__init__(returns_df)
        # Yearly Risk Free Rate = 0.0524 taken from moodle
        # assume 252 trading days.
        self.yearly_rf_rate = 0.0524
        self.rf_rate = self.yearly_rf_rate / 252

        self.assets = self.returns.columns

        self.initial_population = None
        self.final_population = None
        self.optimal_wts = None

        self.delta = delta
        self.eps = eps
        self.cardinality = cardinality

    def enforce_constraints(self, weights: np.ndarray):
        def enforce_cardinality(wts):
            # Replacement minimum hold strategy from Deng, Lin and Lo (2011)
            # cardinality constraint: keep top k (by weight) stocks
            # np.argsort() returns indices of sorted array (ascending)
            # use [:-k] to get indices for all except top k
            sorted_indices = np.argsort(wts)
            top_k = sorted_indices[-self.cardinality :]
            topk_nonzero = np.sum(wts[top_k] > 0)

            # add randomly selected assets (assigned minimum weight) to fulfil cardinality constraint
            if 0 < topk_nonzero < self.cardinality:
                shortfall = self.cardinality - topk_nonzero
                # randomly sample without replacement to add to portfolio shortfall
                random_indices = np.random.choice(
                    sorted_indices[:-topk_nonzero], shortfall, replace=False
                )
                wts[random_indices] = self.eps[random_indices]
            elif topk_nonzero == 0:
                # randomly select top k assets to keep
                random_indices = np.random.choice(
                    sorted_indices, self.cardinality, replace=False
                )
                wts[random_indices] = self.eps[random_indices]
            else:
                # remove the smallest assets to fulfil cardinality constraint
                not_top_k = sorted_indices[: -self.cardinality]
                wts[not_top_k] = 0

            wts /= wts.sum()
            if np.sum(wts != 0) != self.cardinality or np.any(wts < 0):
                print(wts)
                print(np.sum(wts))
                raise ValueError("s")
            return wts

        def enforce_eps_delta(wts, eps_c, delta_c):
            # wts is interpreted as a "value" assigned to the assets as a result of running GA
            # eps_c is the floor constraints of the weights that are non-zero
            # delta_c is the ceiling constraints of the weights that are non-zero.
            # The main loop of handling floor ceiling constraint as described in Chang et. al., (2000)

            if np.sum(eps_c) > 1 or np.sum(delta_c) < 1:
                raise ValueError("Floor and Ceiling Constraints Infeasible!")

            N = len(wts)

            Q = np.ones(N, dtype="int32")  # Mask for the set of all invested assets

            L = np.sum(wts)
            free_prop = 1 - np.sum(eps_c)

            w = eps_c + wts * free_prop / L

            R = np.zeros(N, dtype="int32")

            invested_invalid = Q

            while np.any((w[invested_invalid] > delta_c[invested_invalid])):
                R[w[invested_invalid] > delta_c[invested_invalid]] = 1
                L = np.sum(w[invested_invalid])
                free_prop = 1 - (np.sum(eps_c[invested_invalid]) + np.sum(delta_c[R]))

                w[invested_invalid] = np.max(
                    eps_c, eps_c + wts[invested_invalid] * free_prop / L
                )
                w[R] = delta_c[R]

                invested_invalid = Q & ~R

            return w

        num_assets = len(self.assets)
        if self.cardinality < num_assets:
            weights = enforce_cardinality(weights)

            non_zero_idx = np.where(weights != 0)[0]
            non_zero_wts = weights[non_zero_idx]

            non_zero_wts = enforce_eps_delta(
                non_zero_wts, self.eps[non_zero_idx], self.delta[non_zero_idx]
            )
            weights[non_zero_idx] = non_zero_wts

        else:
            weights = enforce_eps_delta(weights, self.eps, self.delta)

        if np.any((weights < self.eps) & (weights > 0)) or np.any(weights > self.delta):
            print(weights)
            raise ValueError("Floor-ceil Constraints failed to be met.")
        if np.sum(weights != 0) != self.cardinality:
            print(weights)
            raise ValueError("Cardinality Constraints failed to be met.")

        return weights

    def sharpe(self, wts):
        portfolio_sd = np.sqrt(wts @ self.cov_matrix @ wts.T)
        expected_returns = np.dot(wts, self.mean_returns)
        # sharpe_ratio = (expected_returns - self.yearly_rf_rate)/(portfolio_sd)
        sharpe_ratio = (expected_returns - self.rf_rate) / (portfolio_sd)
        return sharpe_ratio

    def initialize_population(self, pop_size):
        init_pop = np.random.dirichlet(
            alpha=np.ones(len(self.assets)),
            size=pop_size,
        )

        for i in range(pop_size):
            init_pop[i] = self.enforce_constraints(init_pop[i])

        self.initial_population = init_pop

        return self.initial_population

    def one_point_crossover(self, p1, p2):
        # One-point Crossover
        crossover_point = np.random.randint(1, len(self.assets))

        o1 = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
        o2 = np.concatenate((p2[:crossover_point], p1[crossover_point:]))

        return o1, o2

    def uniform_crossover(self, p1, p2):
        # Uniform Crossover
        crossover_mask = np.random.randint(low=0, high=2, size=len(self.assets))

        o1 = np.where(crossover_mask, p1, p2)
        o2 = np.where(crossover_mask, p2, p1)

        return o1, o2

    def arithmetic_crossover(self, p1, p2):
        # Arithmetic Crossover from Emanuele Stomeo (2020)
        alpha = np.random.rand()

        o1 = alpha * p1 + (1 - alpha) * p2
        o2 = alpha * p2 + (1 - alpha) * p1

        return o1, o2

    def generateParents_Tournament(self, pop, tournament_size=3):
        def Tournament(candidates, k):
            # A single round of tournament
            best_candidate = candidates[0].copy()

            for i in range(1, k):
                next_candidate = candidates[i]
                if self.sharpe(next_candidate) > self.sharpe(best_candidate):
                    best_candidate = next_candidate.copy()

            return best_candidate

        def TournamentSelection(n, k):
            # n is the number of tournament rounds
            # k is the Tournament Size
            # we will then get n distinct parents or less, due to duplicates when randomly selecting k candidates
            # In Genetic Algorithm, Tournament Selection is called with n = 2 to generate two parents.

            parents = np.zeros(n, dtype="object")
            all_candidate_indices = np.random.choice(
                a=len(pop), size=(n, k), replace=False
            )
            for i in range(n):
                candidate_indices = all_candidate_indices[i, :]
                candidates = pop[candidate_indices]
                parents[i] = Tournament(candidates, k)

            return parents

        parent1, parent2 = TournamentSelection(
            n=2,
            k=tournament_size,
        )

        return parent1, parent2

    def generateParents_Roulette(self, pop, norm_fitness):
        parent_indices = np.random.choice(
            a=len(pop), size=2, p=norm_fitness, replace=False
        )
        parents = pop[parent_indices]

        return parents[0], parents[1]

    def generateOffsprings(self, parent1, parent2, crossover_rate, crossover_str):
        crsvr_method = getattr(self, crossover_str)
        if np.random.rand() < crossover_rate:
            offspring1, offspring2 = crsvr_method(parent1, parent2)
        else:
            offspring1 = np.copy(parent1)
            offspring2 = np.copy(parent2)

        return offspring1, offspring2

    def get_mutated(self, wt1, wt2, mutation_rate, mu=0.5):
        if np.random.rand() < mutation_rate:
            wt1_nonzero_idx = np.where(wt1 != 0)
            wt2_nonzero_idx = np.where(wt2 != 0)

            max_wt1_idx = np.argmax(wt1[wt1_nonzero_idx])
            min_wt1_idx = np.argmin(wt1[wt1_nonzero_idx])

            max_wt2_idx = np.argmax(wt2[wt2_nonzero_idx])
            min_wt2_idx = np.argmin(wt2[wt2_nonzero_idx])

            wt1[min_wt1_idx] *= mu
            wt1[max_wt1_idx] += (1 / mu - 1) * wt1[min_wt1_idx]

            wt2[min_wt2_idx] /= mu
            wt2[max_wt2_idx] -= (1 - mu) * wt2[min_wt2_idx]
            if wt2[max_wt2_idx] < 0:
                wt2[max_wt2_idx] = self.eps[max_wt2_idx]
        return wt1, wt2

    def get_elite_indices(self, fitness, elitism):
        argsort_fitness = np.argsort(fitness)
        elite_indices = argsort_fitness[-(elitism + 1) : -1 : 1]

        return elite_indices

    def solveGA(
        self,
        pop_size=1000,
        tournament_size=3,
        crossover_str="one_point_crossover",
        selection_str="Roulette",
        crossover_rate=0.8,
        mutation_rate=0.03,
        mu=0.5,
        num_generations=3000,
        elite_pct=0.1,
        patience_pct=0.1,
        seed=22000265,
    ):
        # https://www.atlantis-press.com/proceedings/jcis-06/273 used:
        # pop_size = 100,
        # crossover_rate = 1.0,
        # mutation_rate = 0.03
        # num_generations = 3000
        np.random.seed(seed=seed)

        scores = []
        wts = []
        elitism = int(np.round(pop_size * elite_pct))
        elite_list = np.zeros(elitism)
        patience = int(np.round(num_generations * patience_pct))
        pop = self.initialize_population(pop_size)

        best_score = self.sharpe(pop[0])
        best_wts = pop[0]

        times = []  # Runtime history
        start_time = time.time()

        no_improvement = 0
        for generation in range(num_generations):
            # Compute fitnesses
            fitness = np.array([self.sharpe(pop[i]) for i in range(pop_size)])
            norm_fitness = np.exp(fitness) / np.sum(np.exp(fitness))  # Softmax
            max_fitness = fitness.max()
            elite_indices = self.get_elite_indices(
                fitness, elitism
            )  # indices of the best performing individuals of pop
            elite_list = pop[
                elite_indices
            ]  # extract the best, put them into elite_list

            if max_fitness > best_score:
                best_score = max_fitness
                best_wts = pop[np.argmax(fitness)]

            new_pop = np.zeros(pop_size, dtype="object")
            new_pop_count = 0
            while new_pop_count < pop_size - elitism:
                if selection_str == "Tournament":
                    p1, p2 = self.generateParents_Tournament(pop, tournament_size)
                if selection_str == "Roulette":
                    p1, p2 = self.generateParents_Roulette(
                        pop=pop, norm_fitness=norm_fitness
                    )
                c1, c2 = self.generateOffsprings(
                    parent1=p1,
                    parent2=p2,
                    crossover_rate=crossover_rate,
                    crossover_str=crossover_str,
                )

                c1, c2 = self.get_mutated(
                    wt1=c1, wt2=c2, mutation_rate=mutation_rate, mu=mu
                )

                c1 = self.enforce_constraints(c1)

                new_pop[new_pop_count] = c1

                if new_pop_count < pop_size - elitism - 1:
                    c2 = self.enforce_constraints(c2)

                    new_pop[new_pop_count + 1] = c2

                new_pop_count += 2

            for i in range(-1, -elitism - 1, -1):
                new_pop[i] = np.copy(elite_list[i])

            pop = new_pop

            scores.append(best_score)
            wts.append(best_wts)
            if generation >= 1 and np.isclose(scores[-2], scores[-1], atol=1e-12):
                no_improvement += 1

            if no_improvement > patience:
                break

            times.append(time.time() - start_time)
        runtime = time.time() - start_time
        self.optimal_wts = best_wts
        self.final_population = pop.copy()
        return scores, wts, times, runtime


if __name__ == "__main__":
    main()

# # %%
# import scipy
# from scipy.stats import normaltest
# import pandas as pd

# return_df = pd.read_csv("data/return_df.csv")
# return_df = return_df.drop(columns=["Date"])

# for asset in return_df.columns:
#     norm_test = normaltest(return_df[asset])
#     print(norm_test.pvalue)


# # %%
