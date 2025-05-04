# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def main():
    # Sample runner code.
    eps_floor = np.array([0.01] * 101)
    delta_ceil = np.array([0.9] * 101)
    data_wd = "data/return_df.csv"
    df = pd.read_csv(data_wd)
    card = 10
    Genetic_Algorithm = GA(df, eps=eps_floor, delta=delta_ceil, cardinality=card)

    # Tuned parameters
    pop_size = 3000
    num_gens = 500
    trial_count = 5

    crsvr = "uniform_crossover"

    runtime_arr = []
    best_scores_arr = []

    global_best_score = 0
    best_wts = []

    print(f"Running Genetic Algorithm ({crsvr}) ...")
    print(f"eps = {eps_floor}")
    print(f"delta = {delta_ceil}")
    print(
        f"{'Trial':<6} {'Seed':<6} {'Best_Score':<12} {'Time Taken (s)':<14} {'Number of Generations':<100}"
    )
    print("-" * 60)
    for i in range(0, trial_count):
        Genetic_Algorithm = GA(df, eps=eps_floor, delta=delta_ceil, cardinality=card)
        scores, wts, times, runtime = Genetic_Algorithm.solveGA(
            pop_size=pop_size,
            tournament_size=7,
            crossover_str=crsvr,
            crossover_rate=0.9,
            mutation_rate=0.1,
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
            global_best_trial = i + 1
            global_best_time = runtime

        trial_opt_weights = Genetic_Algorithm.optimal_wts

        runtime_arr.append(runtime)
        best_scores_arr.append(best_score)

        print(
            f"{i + 1:<6} {i + 1:<6} {best_score:<12.6f} {runtime:<14.6f} {len(scores):<21} \n{trial_opt_weights} {np.sum(Genetic_Algorithm.optimal_wts)}"
        )

    print("\n\n")

    # ax_runtime.legend(loc = "lower right")
    daily_returns = Genetic_Algorithm.returns
    solution = global_best_solution

    portfolio = dict(zip(daily_returns, solution))
    holdings = {k: v for k, v in portfolio.items() if v > 0}
    sorted_holdings = dict(sorted(holdings.items(), key=lambda x: x[1], reverse=True))
    portfolio_returns = daily_returns @ solution

    portfolio_cum_return = (1 + portfolio_returns).prod() - 1

    Genetic_Algorithm.plot_convergence()
    Genetic_Algorithm.plot_portfolio_composition()
    Genetic_Algorithm.plot_score_statistics()
    Genetic_Algorithm.plot_runtime_analysis()

    print("Summary of GA:")
    print("\nOptimal Portfolio Composition:")
    print("------------------------------")
    for stock, weight in sorted_holdings.items():
        print(f"{stock}: {weight:.4f} ({weight * 100:.2f}%)")
    print("\nPerformance Comparison:")
    print(f"Portfolio Sharpe: {Genetic_Algorithm.sharpe(solution):.4f}")
    print(f"Portfolio Cumulative Return: {portfolio_cum_return:.2%}")
    print(f"Best Trial ({global_best_trial}) Runtime : {global_best_time}")


class snp100_Portfolio:
    def __init__(
        self,
        df,
    ):
        # Return Data
        self.df = df

        if "Date" in self.df.columns:
            self.returns = self.df.drop(columns=["Date"])
        else:
            self.returns = self.df
        self.cov_matrix = self.returns.cov().values
        self.assets = self.returns.columns
        self.mean_returns = np.array(
            [np.mean(self.returns[a]) for a in self.returns.columns]
        )


class GA(snp100_Portfolio):
    def __init__(
        self,
        df,
        eps=np.array([0.01] * 101),  # Epsilon is floor constraint
        delta=np.array([0.9] * 101),  # delta is ceiling constraint
        cardinality=10,
    ):
        super().__init__(df)
        # Yearly Risk Free Rate = 0.0524 taken from moodle
        # assume 252 trading days.
        self.yearly_rf_rate = 0.0524
        self.rf_rate = self.yearly_rf_rate / 252

        self.assets = self.returns.columns

        self.initial_population = None
        self.optimal_wts = None

        self.delta = delta
        self.eps = eps
        self.cardinality = cardinality

        # Data for Visualisation
        self.convergence_history = []
        self.mean_generational_sharpe_history = []
        self.median_generational_sharpe_history = []
        self.min_generational_sharpe_history = []
        self.max_generational_sharpe_history = []
        self.std_generational_sharpe_history = []
        self.runtime_history = []

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
                raise ValueError("Cardinality constraint not handled properly.")
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
        # Random initialization of population
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
            # Search for best candidate by sharpe
            best_candidate = candidates[0].copy()

            for i in range(1, k):
                next_candidate = candidates[i]
                if self.sharpe(next_candidate) > self.sharpe(best_candidate):
                    best_candidate = next_candidate.copy()

            return best_candidate

        def TournamentSelection(n, k):
            # n is the number of tournament rounds
            # k is the Tournament Size (selection pressure, Miller et. al., 1995)
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
        # Roulette Selection without replacement.
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
        # GM-mu mutation by Emanuele Stomeo et al., 2020
        # wt1 : decrease min by a factor of mu
        #       increase max such that the wt1 still sums to 1
        #       i.e., the smallest gene is decreased by a factor mu, the largest gene increased
        # wt2 : increase min by a factor of mu
        #       decrease max such that the wt1 still sums to 1
        #       i.e., the smalest gene is increased by a factor of mu, the smallest gene decreased
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
                wt2[max_wt2_idx] = self.eps[
                    max_wt2_idx
                ]  # move to eps when decreased too much. Invoke repair algorithm after mutation
        return wt1, wt2

    def get_elite_indices(self, fitness, elitism):
        argsort_fitness = np.argsort(fitness)
        elite_indices = argsort_fitness[-(elitism + 1) : -1 : 1]

        return elite_indices

    def solveGA(
        self,
        pop_size=3000,
        tournament_size=7,
        crossover_str="uniform_crossover",
        crossover_rate=0.9,
        mutation_rate=0.1,
        mu=0.4632970696461231,
        num_generations=500,
        elite_pct=0.1,
        patience_pct=0.2,
        seed=22000265,
    ):
        """

        Args:
            pop_size (int, optional): (positive int)
                population size of GA. Defaults to 1000.

            tournament_size (int, optional): (positive int)
                No. of individuals per tournament during tournament selection. Defaults to 3.
                set tournament_size = 0 for roulette wheel selection

            crossover_str (str, optional): {"one_point_crossover", "uniform_crossover", "arithmetic_crossover"}
                Crossover method. Defaults to "one_point_crossover".

            crossover_rate (float, optional): [0, 1]
                probability of crossover for each pair of parents (no crossover means offsprings are direct copies). Defaults to 0.8.

            mutation_rate (float, optional): [0, 1]
                probability of mutation for each pair of children. Defaults to 0.03.

            mu (float, optional): [0, 1]
                mu in [0, 1]. Reflects the intensity of the GM-mu mutation. Defaults to 0.5.

            num_generations (int, optional): (positive int)
                number of generations. Defaults to 3000.

            elite_pct (float, optional): [0, 1]
                Percentage of offsprings to be considered elites. Defaults to 0.1.

            patience_pct (float, optional): [0, 1]
                Number of generations of no improvement before stopping. Set to 1 for no early stop. Defaults to 0.1.
                Tolerance for no improvement is set to 1e-12.

            seed (int, optional):
                seed for reproducibility. Defaults to 22000265.

        Returns:
            convergence_history : Record of best sharpe ratio over each generation
            wts : record of weights corresponding to best sharpe ratio over each generation
            times : time taken for the ith generation
            runtime : Time taken for solveGA to optimise portfolio
        """
        np.random.seed(seed=seed)

        wts = []
        elitism = int(np.round(pop_size * elite_pct))
        elite_list = np.zeros(elitism)
        patience = int(np.round(num_generations * patience_pct))
        pop = self.initialize_population(pop_size)

        best_score = self.sharpe(pop[0])
        best_wts = pop[0]

        times = []  # Runtime history

        start_time = time.time()
        total_runtime = 0.0
        no_improvement = 0

        for generation in range(num_generations):
            # Compute fitnesses
            fitness = np.array([self.sharpe(pop[i]) for i in range(pop_size)])
            self.mean_generational_sharpe_history.append(np.mean(fitness))
            self.median_generational_sharpe_history.append(np.median(fitness))
            self.min_generational_sharpe_history.append(np.min(fitness))
            self.max_generational_sharpe_history.append(np.max(fitness))
            self.std_generational_sharpe_history.append(np.std(fitness))
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
            self.convergence_history.append(best_score)
            new_pop = np.zeros(pop_size, dtype="object")
            new_pop_count = 0
            while new_pop_count < pop_size - elitism:
                if tournament_size != 0:
                    p1, p2 = self.generateParents_Tournament(pop, tournament_size)
                else:
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

            wts.append(best_wts)
            if generation >= 1 and np.isclose(
                self.convergence_history[-2], self.convergence_history[-1], atol=1e-12
            ):
                no_improvement += 1

            gen_time = time.time() - start_time - total_runtime
            total_runtime = time.time() - start_time

            times.append(gen_time)
            self.runtime_history.append(total_runtime)

            if no_improvement > patience:
                break

        runtime = time.time() - start_time
        self.optimal_wts = best_wts

        self.runtime_data = {
            "total": runtime,
            "per_iteration": times,
            "cumulative": self.runtime_history,
        }

        return self.convergence_history, wts, times, runtime

    # Visualisation functions
    def plot_convergence(self):
        """Plot the convergence history of the optimization process and return the figure"""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        iterations = range(len(self.mean_generational_sharpe_history))
        avg_scores = np.array(self.mean_generational_sharpe_history)
        std_scores = np.array(self.std_generational_sharpe_history)

        ax1.plot(
            iterations,
            avg_scores,
            label="Mean Generational Sharpe",
            color="green",
            linestyle="--",
        )
        ax1.fill_between(
            iterations,
            avg_scores - std_scores,
            avg_scores + std_scores,
            color="green",
            alpha=0.2,
            label="±1 Std Dev",
        )

        ax1.plot(
            iterations,
            self.convergence_history,
            color="purple",
            alpha=0.5,
            linestyle="-",
            label="Global Best Sharpe",
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.set_title("GA Convergence History")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        fig.tight_layout()

        plt.show()

        return fig

    def plot_runtime_analysis(self):
        """Plot runtime analysis of the optimization process and return the figure"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.runtime_data["cumulative"], color="blue", linewidth=2)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cumulative Runtime (seconds)")
        ax1.set_title("Cumulative Runtime vs Iterations")
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.runtime_data["per_iteration"], color="green", linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Time per Iteration (seconds)")
        ax2.set_title("Time per Iteration")
        ax2.grid(True, alpha=0.3)

        # Add moving average
        window_size = min(10, len(self.runtime_data["per_iteration"]))
        if window_size > 1:
            moving_avg = np.convolve(
                self.runtime_data["per_iteration"],
                np.ones(window_size) / window_size,
                mode="valid",
            )
            ax2.plot(
                range(window_size - 1, len(self.runtime_data["per_iteration"])),
                moving_avg,
                color="red",
                linewidth=1.5,
                linestyle="--",
                label=f"{window_size}-iter Moving Avg",
            )
            ax2.legend()

        fig.tight_layout()

        plt.show()

        return fig

    def plot_score_statistics(self):
        """Plot statistical analysis of scores across iterations and return the figure"""
        iterations = range(len(self.mean_generational_sharpe_history))

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            iterations,
            self.mean_generational_sharpe_history,
            label="Mean Score",
            color="green",
            linewidth=2,
        )
        ax.plot(
            iterations,
            self.median_generational_sharpe_history,
            label="Median Score",
            color="blue",
            linestyle="--",
        )
        ax.plot(
            iterations,
            self.max_generational_sharpe_history,
            label="Max Score",
            color="purple",
        )
        ax.plot(
            iterations,
            self.min_generational_sharpe_history,
            label="Min Score",
            color="red",
        )

        ax.fill_between(
            iterations,
            self.min_generational_sharpe_history,
            self.max_generational_sharpe_history,
            color="lightblue",
            alpha=0.3,
            label="Score Range",
        )

        interval = max(1, len(iterations) // 20)
        selected_iterations = list(iterations)[::interval]
        selected_means = [
            self.mean_generational_sharpe_history[i] for i in selected_iterations
        ]
        selected_stds = [
            self.std_generational_sharpe_history[i] for i in selected_iterations
        ]

        ax.errorbar(
            selected_iterations,
            selected_means,
            yerr=selected_stds,
            fmt="o",
            color="darkgreen",
            capsize=5,
            label="Std Dev",
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Score Statistics Across Iterations")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()

        plt.show()

        return fig

    def plot_portfolio_composition(self):
        """Plot the composition of the optimal portfolio and return the figure"""
        weights = self.optimal_wts
        assets = self.returns.columns

        non_zero_indices = np.where(weights > 0)[0]
        non_zero_weights = weights[non_zero_indices]
        non_zero_assets = [assets[i] for i in non_zero_indices]

        sorted_indices = np.argsort(non_zero_weights)[::-1]
        sorted_weights = non_zero_weights[sorted_indices]
        sorted_assets = [non_zero_assets[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(sorted_assets)), sorted_weights, color="skyblue")
        ax.set_xticks(range(len(sorted_assets)))
        ax.set_xticklabels(sorted_assets, rotation=45, ha="right")
        ax.set_title(
            f"Optimal Portfolio Composition (Sharpe: {self.sharpe(self.optimal_wts):.4f})"
        )
        ax.set_xlabel("Assets")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3, axis="y")

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{sorted_weights[i]:.3f}",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()

        plt.show()

        return fig


# %%
if __name__ == "__main__":
    main()
# %%
