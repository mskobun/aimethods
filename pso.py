from typing import Callable
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

"""
Inspired by the approach in:
A Hybrid Algorithm of Particle Swarm Optimization and Tabu Search for Distribution Network Reconfiguration,
Fang and Zhang (2016)

When gbest stagnates for 50 iterations, which may be a sign of early convergence,
the original swarm splits into 3 swarms: 
1. Swarm 1 continues performing the original PSO.
2. Swarm 2 is reset with new random positions and velocities. 
3. Swarm 3 performs tabu search based on the current optimal solution.
The best solution from all 3 swarms is selected as the new global best. 
The regular PSO resumes until stopping criteria are met or maximum iterations are reached.
"""


def main():
    returns = pd.read_csv("data/return_df.csv", index_col=0)  # column 0 is date
    risk_free_rate = 0.0524 / 365

    # Set seed for reproducibility
    np.random.seed(42)
    pso = PSO(returns, risk_free_rate=risk_free_rate)
    solution, score, runtime_data = pso.run(500)

    portfolio = dict(zip(returns.columns, solution))
    holdings = {k: v for k, v in portfolio.items() if v > 0}
    sorted_holdings = dict(sorted(holdings.items(), key=lambda x: x[1], reverse=True))

    pso.plot_convergence()
    pso.plot_runtime_analysis(runtime_data)
    pso.plot_score_statistics()
    pso.plot_portfolio_composition()

    print("\nOptimal Portfolio Composition:")
    print("------------------------------")
    for stock, weight in sorted_holdings.items():
        print(f"{stock}: {weight:.4f} ({weight * 100:.2f}%)")
    print(f"Portfolio Sharpe: {score:.4f}")


class Particle:
    """
    Each particle is a portfolio of n stocks (n = total number of investable stocks).
    """

    def __init__(self, investable_size: int, weight_min: float, weight_max: float):
        # Position vector of the particle
        # Generate uniform random numbers in [weight_min,weight_max)
        self.position = np.random.uniform(weight_min, weight_max, investable_size)
        self.position /= np.sum(self.position)  # normalize weights to sum to 1
        # Velocity vector of the particle
        # Generate random numbers from a standard normal distribution (mean 0, std 1) then scaled by 0.1
        self.velocity = np.random.standard_normal(investable_size) * 0.1
        # Personal best obtained so far by the particle
        self.personal_best = self.position.copy()  # initialize at random position
        # Fitness value of the personal best
        self.personal_best_sharpe = -np.inf
        self.patience_reflection = 0
        self.last_personal_best_sharpe = -np.inf


class PSO:
    def __init__(
        self,
        returns: pd.DataFrame,
        particles_count: int = 200,
        portfolio_size: int = 10,
        weight_max: float = 0.9,
        weight_min: float = 0.01,
        risk_free_rate: float = 0.0524 / 365,
        quiet: bool = False,
    ):
        self.validate_constraints(portfolio_size, weight_min, weight_max)
        self.returns = returns
        self.investable_size = returns.shape[1]
        self.portfolio_size = portfolio_size
        self.weight_max = weight_max
        self.weight_min = weight_min
        self.particles_count = particles_count
        self.global_best = None
        self.global_best_sharpe = -np.inf
        self.mean_return = returns.mean()
        self.covariance_matrix = returns.cov()
        self.risk_free_rate = risk_free_rate
        self.quiet = quiet
        # For visualization
        self.convergence_history = []
        self.avg_personal_best_history = []
        self.runtime_history = []
        self.iteration_times = []
        self.min_scores_history = []
        self.max_scores_history = []
        self.median_scores_history = []
        self.std_scores_history = []

    def evaluate(self, weights: np.ndarray):
        # Calculate portfolio returns
        portfolio_return = np.sum(self.mean_return * weights)
        # Calculate portfolio standard deviation
        portfolio_sd = np.sqrt(
            np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        )
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_sd
        return sharpe_ratio

    def validate_constraints(
        self, portfolio_size: int, weight_min: float, weight_max: float
    ):
        if weight_min * portfolio_size > 1.0:
            original_min = weight_min
            weight_min = 1.0 / portfolio_size  # Adjust to the maximum possible value
            print(
                f"\nWARNING: Minimum weight constraint ({original_min:.2f}) conflicts with portfolio size ({portfolio_size})."
            )
            print(
                f"The minimum weight has been automatically adjusted to {weight_min:.4f} to ensure mathematical feasibility."
            )
            print(
                f"With {portfolio_size} assets, the maximum possible minimum weight is {1.0 / portfolio_size:.4f}."
            )
            print(
                "Consider either decreasing the minimum weight or the portfolio size."
            )

        if weight_max < weight_min:
            weight_max = max(weight_min, 1.0)
            print(
                f"\nWARNING: Maximum weight constraint ({weight_max:.2f}) was less than minimum weight ({weight_min:.2f})."
            )
            print(
                f"The maximum weight has been automatically adjusted to {weight_max:.4f}."
            )

    def check_stagnation(self, history, window=50):
        if len(history) < window:
            return False

        # Check if last 50 values of gbest are the same when rounded to 3 decimal places (view in the console)
        # This avoids floating point precision issues. It is also a more aggressive approach to force swarm splitting.
        recent_global = history[-window:]
        rounded_values = [round(value, 3) for value in recent_global]
        first_rounded_value = rounded_values[0]

        # Check if all rounded values are the same
        for rounded_value in rounded_values:
            if rounded_value != first_rounded_value:
                return False

        return True

    def _handle_cardinality(self, particle):
        # Replacement minimum hold strategy from Deng, Lin and Lo (2011)
        # cardinality constraint: keep top k (by weight) stocks
        # np.argsort() returns indices of sorted array (ascending)
        # use [:-k] to get indices for all except top k
        sorted_indices = np.argsort(particle.position)
        top_k = sorted_indices[-self.portfolio_size :]
        topk_nonzero = np.sum(particle.position[top_k] > 0)

        # add randomly selected assets (assigned minimum weight) to fulfil cardinality constraint
        if 0 < topk_nonzero < self.portfolio_size:
            shortfall = self.portfolio_size - topk_nonzero
            # randomly sample without replacement to add to portfolio shortfall
            random_indices = np.random.choice(
                sorted_indices[:-topk_nonzero], shortfall, replace=False
            )
            particle.position[random_indices] = self.weight_min
        elif topk_nonzero == 0:
            # randomly select top k assets to keep
            random_indices = np.random.choice(
                sorted_indices, self.portfolio_size, replace=False
            )
            particle.position[random_indices] = self.weight_min
        else:
            # remove the smallest assets to fulfil cardinality constraint
            not_top_k = sorted_indices[: -self.portfolio_size]
            particle.position[not_top_k] = 0

        particle.position /= np.sum(particle.position)  # normalize weights

    def _apply_reflection(self, particle, patience_reflection_limit=20):
        # Reflection strategy from Paterlini and Krink (2006)
        # to help individual particles escape local optima and explore more of the search space
        # the strategy terminates when no improvement for the particle after a certain number of iterations.
        # Patience reflection limit is set to 20 iterations (Deng, Lin and Lo, 2011)
        if (
            particle.personal_best_sharpe <= particle.last_personal_best_sharpe
            and particle.patience_reflection > patience_reflection_limit
        ):
            particle.patience_reflection = 0
            # ensure non-negative and ceiling constraint
            particle.position = np.clip(particle.position, 0, self.weight_max)

            # apply floor constraint only for non-zero weights
            particle.position[particle.position > 0] = np.maximum(
                self.weight_min, particle.position[particle.position > 0]
            )
        else:  # reflection strategy
            particle.patience_reflection += 1

            # handle negative values
            particle.position[particle.position < 0] = 0

            # handle ceiling constraint
            ceiling_mask = particle.position > self.weight_max
            copy_invalid_ceiling = particle.position[
                ceiling_mask
            ]  # new array to avoid modifying original
            reflected_ceiling = copy_invalid_ceiling - 2 * (
                copy_invalid_ceiling - self.weight_min
            )
            particle.position[ceiling_mask] = np.maximum(
                0, reflected_ceiling
            )  # reflect while maintaining non-negativity

            # handle floor constraint
            # use & for element-wise logical AND with numpy arrays
            floor_mask = (particle.position < self.weight_min) & (
                particle.position > 0
            )  # apply only to non-zero weights
            copy_invalid_floor = particle.position[floor_mask]
            reflected_floor = copy_invalid_floor + 2 * (
                self.weight_min - copy_invalid_floor
            )
            particle.position[floor_mask] = np.minimum(
                self.weight_max, reflected_floor
            )  # reflect while maintaining ceiling constraint

        particle.last_personal_best_sharpe = particle.personal_best_sharpe

        particle.position /= np.sum(particle.position)  # normalize weights

    def _iterate_pso(self, i, iterations, w_max, w_min, c1_max, c1_min, c2_max, c2_min):
        """Perform one iteration of PSO"""
        for particle in self.particles:
            # use single random value for all weights in a position vector
            random_1 = np.random.random()
            random_2 = np.random.random()
            # linear decreasing
            inertia_weight = (w_min - w_max) * i / iterations + w_max
            # linear decreasing
            cognitive_weight = (c1_min - c1_max) * i / iterations + c1_max
            # linear increasing
            social_weight = (c2_max - c2_min) * i / iterations + c2_min
            # use broadcasting in numpy for element-wise multiplication of scalar (the weights) with particle.velocity (np.array)
            particle.velocity = (
                inertia_weight * particle.velocity
                + cognitive_weight
                * random_1
                * (particle.personal_best - particle.position)
                + social_weight * random_2 * (self.global_best - particle.position)
            )
            particle.position += particle.velocity

            self.mutation(particle, i, iterations)
            self._handle_cardinality(particle)
            self._apply_reflection(particle)
            particle.position = self.enforce_floor_constraint(particle.position)

            # Update personal best
            result = self.evaluate(particle.position)
            if result > particle.personal_best_sharpe:
                particle.personal_best_sharpe = result
                particle.personal_best = particle.position.copy()

                # Update global best
                if result > self.global_best_sharpe:
                    self.global_best_sharpe = result
                    self.global_best = particle.position.copy()
                    self.global_best = self.enforce_floor_constraint(self.global_best)

    def _continue_current_swarm(self, iterations):
        """Continue with 1/3 of the current swarm"""
        current_particles = [
            Particle(self.investable_size, self.weight_min, self.weight_max)
            for _ in range(self.particles_count // 3)
        ]
        # copy the first 1/3 of the particles to the new swarm
        # zip will stop at the shorter iterable (current_particles)
        for p, old_p in zip(current_particles, self.particles):
            p.position = old_p.position.copy()
            p.velocity = old_p.velocity.copy()

        temp_pso = PSO(self.returns, quiet=self.quiet)
        temp_pso.particles = current_particles
        return temp_pso.run(iterations)

    def _reset_and_continue(self, iterations):
        """Reset 1/3 of the swarm with new random positions and velocities"""
        temp_pso = PSO(self.returns, self.particles_count // 3, quiet=self.quiet)
        return temp_pso.run(iterations)

    def run(self, iterations: int):
        # Initialise particles
        self.particles = [
            Particle(self.investable_size, self.weight_min, self.weight_max)
            for _ in range(self.particles_count)
        ]
        # Initialise global best to the first particle (or the best among all pbest)
        self.global_best = self.particles[0].position

        global_best_history = []
        best_solution = None
        best_score = -np.inf

        # Use recommended values from Deng, Lin and Lo (2011)
        # as suggested in Engelbrecht (2005), Ratnaweera et al. (2004), and Tripathi et al. (2007).
        w_max = 0.9
        w_min = 0.4
        c1_max = c2_max = 2.5
        c1_min = c2_min = 0.5

        patience_global_limit = 100  # value from Deng, Lin and Lo (2011)
        patience_global = 0

        start_time = time.time()
        total_runtime = 0.0
        i = 0

        # Reset visualization data
        self.convergence_history = []
        self.avg_personal_best_history = []
        self.runtime_history = []
        self.iteration_times = []
        self.min_scores_history = []
        self.max_scores_history = []
        self.median_scores_history = []
        self.std_scores_history = []

        while i < iterations:
            iteration_start = time.time()
            last_best_sharpe = self.global_best_sharpe

            # One iteration of regular PSO
            self._iterate_pso(
                i, iterations, w_max, w_min, c1_max, c1_min, c2_max, c2_min
            )

            # Collect score statistics for this iteration
            current_scores = np.array([p.personal_best_sharpe for p in self.particles])

            # Save history for visualization
            self.convergence_history.append(self.global_best_sharpe)
            self.avg_personal_best_history.append(
                np.mean([p.personal_best_sharpe for p in self.particles])
            )
            self.min_scores_history.append(np.min(current_scores))
            self.max_scores_history.append(np.max(current_scores))
            self.median_scores_history.append(np.median(current_scores))
            self.std_scores_history.append(np.std(current_scores))

            iteration_end = time.time()
            iteration_time = iteration_end - iteration_start
            self.iteration_times.append(iteration_time)

            total_runtime += iteration_time
            self.runtime_history.append(total_runtime)

            global_best_history.append(self.global_best_sharpe)

            # Check for stagnation
            if self.check_stagnation(global_best_history):
                if not self.quiet:
                    print("Stagnation detected - splitting swarms")

                # Store current best solution
                current_best = self.global_best.copy()
                current_best_score = self.global_best_sharpe

                # Swarm 1: Continue current algorithm with 1/3 of existing particles
                swarm1_solution, swarm1_score, _ = self._continue_current_swarm(50)

                # Swarm 2: Reset with new positions and velocities, and continue
                swarm2_solution, swarm2_score, _ = self._reset_and_continue(50)

                # Swarm 3: Tabu search using current best solution
                tabu = Tabu(
                    investable_size=self.investable_size,
                    initial_solution=current_best,
                    initial_sharpe=current_best_score,
                    iterations=50,
                    portfolio_size=self.portfolio_size,
                    weight_min=self.weight_min,
                    weight_max=self.weight_max,
                    evaluate=lambda weights: self.evaluate(weights),
                    quiet=self.quiet,
                )
                swarm3_solution, swarm3_score, _ = tabu.run()

                # Select best solution from the three swarms
                solutions = [
                    (swarm1_solution, swarm1_score),
                    (swarm2_solution, swarm2_score),
                    (swarm3_solution, swarm3_score),
                ]
                best_solution, best_score = max(solutions, key=lambda x: x[1])

                # Update global best with the best solution
                self.global_best = best_solution.copy()
                self.global_best_sharpe = best_score

                # Reset history
                global_best_history = []

                if not self.quiet:
                    print(
                        f"Swarms returned Best Sharpe ratio = {round(self.global_best_sharpe, 4)}"
                    )
                    print("Resume normal PSO")
                    print("---")

            patience_global = (
                patience_global + 1
                if self.global_best_sharpe <= last_best_sharpe
                else 0
            )

            # early stopping condition
            if patience_global > patience_global_limit:
                break

            i += 1

        runtime = time.time() - start_time
        if not self.quiet:
            print(f"Runtime: {round(runtime, 4)}")
            print(f"Optimal Sharpe ratio: {round(self.global_best_sharpe, 4)}")
            print(f"Optimal weights: {self.global_best}")
            print(f"{np.sum(self.global_best)}")

        # Compile runtime data for return
        runtime_data = {
            "total": runtime,
            "per_iteration": self.iteration_times,
            "cumulative": self.runtime_history,
        }

        return self.global_best, self.global_best_sharpe, runtime_data

    def enforce_floor_constraint(self, position):
        below_min_mask = (0 < position) & (position < self.weight_min)
        if np.any(below_min_mask):
            # Set weights below minimum to the minimum
            position[below_min_mask] = self.weight_min

            # Reduce other weights proportionally to maintain sum=1
            above_min_mask = position > self.weight_min
            if np.any(above_min_mask):
                # Calculate excess weight after setting below-min weights to minimum
                excess = np.sum(position) - 1.0
                # Scale down above-min weights proportionally
                position[above_min_mask] *= 1.0 - excess / np.sum(
                    position[above_min_mask]
                )

            # Final normalization to ensure sum=1 exactly
            position /= np.sum(position)
        return position

    def mutation(self, particle, i, iterations):
        mutation_dependence = 5

        # To maximise diversity, use mutation operator based on Tripathi et al., 2007
        # Given a particle, a randomly chosen variable is mutated
        def mutate(i, x):
            r = np.random.random()
            return x * (1 - r ** ((1 - i / iterations) ** mutation_dependence))

        mutation_index = np.random.randint(0, self.investable_size)
        flip = np.random.randint(0, 2)
        if flip == 0:
            mutated_weight = particle.position[mutation_index] + mutate(
                i, self.weight_max - particle.position[mutation_index]
            )
            particle.position[mutation_index] = np.maximum(0, mutated_weight)
        else:  # flip == 1
            mutated_weight = particle.position[mutation_index] - mutate(
                i, particle.position[mutation_index] - self.weight_min
            )
            particle.position[mutation_index] = np.maximum(0, mutated_weight)

        particle.position /= np.sum(particle.position)  # normalize weights

    # Visualization methods
    def plot_convergence(self):
        """Plot the convergence history of the optimization process"""
        plt.figure(figsize=(12, 6))

        # Create primary y-axis for Sharpe ratios
        ax1 = plt.gca()
        ax1.plot(
            self.convergence_history,
            label="Global Best Sharpe",
            color="blue",
            linewidth=2,
        )
        ax1.plot(
            self.avg_personal_best_history,
            label="Average Personal Best",
            color="green",
            linestyle="--",
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.grid(True, alpha=0.3)

        # Add standard deviation band around average
        iterations = range(len(self.avg_personal_best_history))
        avg_scores = np.array(self.avg_personal_best_history)
        std_scores = np.array(self.std_scores_history)
        ax1.fill_between(
            iterations,
            avg_scores - std_scores,
            avg_scores + std_scores,
            color="green",
            alpha=0.2,
            label="±1 Std Dev",
        )

        # Add min/max bounds
        ax1.plot(
            self.min_scores_history,
            color="red",
            alpha=0.5,
            linestyle=":",
            label="Min Score",
        )
        ax1.plot(
            self.max_scores_history,
            color="purple",
            alpha=0.5,
            linestyle=":",
            label="Max Score",
        )

        plt.title("PSO Convergence History")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_runtime_analysis(self, runtime_data):
        """Plot runtime analysis of the optimization process"""
        # plt.figure(figsize=(12, 6))

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Cumulative runtime
        ax1.plot(runtime_data["cumulative"], color="blue", linewidth=2)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cumulative Runtime (seconds)")
        ax1.set_title("Cumulative Runtime vs Iterations")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Per-iteration runtime
        ax2.plot(runtime_data["per_iteration"], color="green", linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Time per Iteration (seconds)")
        ax2.set_title("Time per Iteration")
        ax2.grid(True, alpha=0.3)

        # Add moving average to smooth the per-iteration plot
        window_size = min(10, len(runtime_data["per_iteration"]))
        if window_size > 1:
            moving_avg = np.convolve(
                runtime_data["per_iteration"],
                np.ones(window_size) / window_size,
                mode="valid",
            )
            ax2.plot(
                range(window_size - 1, len(runtime_data["per_iteration"])),
                moving_avg,
                color="red",
                linewidth=1.5,
                linestyle="--",
                label=f"{window_size}-iter Moving Avg",
            )
            ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_score_statistics(self):
        """Plot statistical analysis of scores across iterations"""
        plt.figure(figsize=(12, 6))

        iterations = range(len(self.avg_personal_best_history))

        plt.plot(
            iterations,
            self.avg_personal_best_history,
            label="Mean Score",
            color="green",
            linewidth=2,
        )
        plt.plot(
            iterations,
            self.median_scores_history,
            label="Median Score",
            color="blue",
            linestyle="--",
        )
        plt.plot(iterations, self.max_scores_history, label="Max Score", color="purple")
        plt.plot(iterations, self.min_scores_history, label="Min Score", color="red")

        # Fill area between min and max for range visualization
        plt.fill_between(
            iterations,
            self.min_scores_history,
            self.max_scores_history,
            color="lightblue",
            alpha=0.3,
            label="Score Range",
        )

        # Plot standard deviation as error bars at regular intervals
        interval = max(1, len(iterations) // 20)  # Show at most 20 error bars
        selected_iterations = iterations[::interval]
        selected_means = [
            self.avg_personal_best_history[i] for i in selected_iterations
        ]
        selected_stds = [self.std_scores_history[i] for i in selected_iterations]

        plt.errorbar(
            selected_iterations,
            selected_means,
            yerr=selected_stds,
            fmt="o",
            color="darkgreen",
            capsize=5,
            label="Std Dev",
        )

        plt.xlabel("Iteration")
        plt.ylabel("Sharpe Ratio")
        plt.title("Score Statistics Across Iterations")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_portfolio_composition(self):
        """Plot the composition of the optimal portfolio"""
        # Get non-zero weights and corresponding asset names
        weights = self.global_best
        assets = self.returns.columns

        # Filter for assets with non-zero weights and sort by weight
        non_zero_indices = np.where(weights > 0)[0]
        non_zero_weights = weights[non_zero_indices]
        non_zero_assets = [assets[i] for i in non_zero_indices]

        # Sort by weight (descending)
        sorted_indices = np.argsort(non_zero_weights)[::-1]
        sorted_weights = non_zero_weights[sorted_indices]
        sorted_assets = [non_zero_assets[i] for i in sorted_indices]

        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(sorted_assets)), sorted_weights, color="skyblue")
        plt.xticks(range(len(sorted_assets)), sorted_assets, rotation=45, ha="right")
        plt.title(
            f"Optimal Portfolio Composition (Sharpe: {self.global_best_sharpe:.4f})"
        )
        plt.xlabel("Assets")
        plt.ylabel("Weight")
        plt.grid(True, alpha=0.3, axis="y")

        # Add weight values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{sorted_weights[i]:.3f}",
                ha="center",
                va="bottom",
                rotation=0,
            )

        plt.tight_layout()
        plt.show()


class Tabu:
    def __init__(
        self,
        investable_size: int,
        initial_solution: np.ndarray,
        initial_sharpe: float = 0,
        iterations: int = 100,
        portfolio_size: int = 10,
        weight_min: float = 0.01,
        weight_max: float = 0.9,
        evaluate: Callable[[np.ndarray], float] = None,
        quiet: bool = False,
    ):
        if initial_solution is None:
            raise ValueError(
                "An initial solution from PSO must be passed into Tabu search in this hybrid implementation."
            )

        self.current_solution = initial_solution
        self.current_sharpe = initial_sharpe
        self.optimal_solution = self.current_solution.copy()
        self.optimal_sharpe = self.current_sharpe
        self.iterations = iterations
        self.investable_size = investable_size
        self.portfolio_size = portfolio_size
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.evaluate = evaluate
        self.quiet = quiet

        # Tabu list: each row is a stock with two columns:
        # Column 0: Tabu time for increase move. Initialized to 0 (non-tabu)
        # Column 1: Tabu time for decrease move. Initialized to 0 (non-tabu).
        self.tabu_list = np.zeros((investable_size, 2))

        # From Chang et al (2000)
        self.tabu_tenure = 7
        self.increase = 1.1
        self.decrease = 0.9

    def increase_move(self, solution: np.ndarray, stock: int):
        neighbour = solution.copy()
        new_weight = (
            self.increase * (self.weight_min + neighbour[stock]) - self.weight_min
        )
        neighbour[stock] = np.minimum(new_weight, self.weight_max)
        neighbour /= np.sum(neighbour)
        return neighbour

    def decrease_move(self, solution: np.ndarray, stock: int):
        neighbour = solution.copy()
        new_weight = (
            self.decrease * (self.weight_min + neighbour[stock]) - self.weight_min
        )
        if new_weight < 0:
            # replace with a randomly selected stock which was not in the portfolio i.e. weight = 0
            # but due to floating point arithmetic, it may not be exactly 0 so use np.isclose
            not_in_portfolio = np.where(np.isclose(neighbour, 0))[0]
            if len(not_in_portfolio) > 0:
                random_stock = np.random.choice(not_in_portfolio)
                neighbour[random_stock] = self.weight_min
                neighbour[stock] = 0  # remove from portfolio
            else:
                neighbour[stock] = self.weight_min
        else:
            neighbour[stock] = np.maximum(new_weight, self.weight_min)

        neighbour /= np.sum(neighbour)
        return neighbour

    def run(self):
        patience = 0
        patience_limit = 10
        start_time = time.time()

        for i in range(self.iterations):
            best_neighbour = None
            best_neighbour_score = -np.inf
            best_neighbour_stock = None

            # get portfolio
            sorted_indices = np.argsort(self.current_solution)
            portfolio = sorted_indices[-self.portfolio_size :]

            # generate a neighbour by increasing or decreasing the weight of a stock in portfolio
            for stock in portfolio:
                increase_neighbour = self.increase_move(self.current_solution, stock)
                decrease_neighbour = self.decrease_move(self.current_solution, stock)
                increase_score = self.evaluate(increase_neighbour)
                decrease_score = self.evaluate(decrease_neighbour)
                # choose the better option between increase and decrease
                neighbour = (
                    increase_neighbour
                    if increase_score > decrease_score
                    else decrease_neighbour
                )
                score = (
                    increase_score
                    if increase_score > decrease_score
                    else decrease_score
                )
                move = 0 if increase_score > decrease_score else 1

                # if move is not tabu or aspiration criterion is met
                if (self.tabu_list[stock][move] <= i) or (score > self.optimal_sharpe):
                    if score > best_neighbour_score:
                        best_neighbour = neighbour.copy()
                        best_neighbour_score = score
                        best_neighbour_stock = stock

            # update the solution and tabu list
            if best_neighbour is not None:
                self.current_solution = best_neighbour.copy()
                self.current_sharpe = best_neighbour_score

                # From Chang et al (2000), update tabu time for both increase and decrease moves, not just the best move
                self.tabu_list[best_neighbour_stock][0] = i + self.tabu_tenure
                self.tabu_list[best_neighbour_stock][1] = i + self.tabu_tenure

                if best_neighbour_score > self.optimal_sharpe:
                    self.optimal_solution = best_neighbour.copy()
                    self.optimal_sharpe = best_neighbour_score
                    patience = 0
                else:
                    patience += 1

            if patience >= patience_limit:
                break

        runtime = time.time() - start_time
        if not self.quiet:
            print(f"Tabu Runtime: {round(runtime, 4)}")
            print(f"Tabu Optimal Sharpe ratio: {round(self.optimal_sharpe, 4)}")
            print(f"Tabu Optimal weights: {self.optimal_solution}")
            print(f"{np.sum(self.optimal_solution)}")

        return self.optimal_solution, self.optimal_sharpe, runtime


if __name__ == "__main__":
    main()
