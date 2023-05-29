import deap.tools
import numpy as np
from math import floor
from deap import base, creator, tools, algorithms
import random

from typing import List, Tuple
import time

random.seed(42)
np.random.seed(42)

TRADE_HISTORY = True


class StockTradingSimulation:
    def __init__(self):
        self.population_size: int = 1000
        self.num_generations: int = 10000

        self.initial_investment = 10000.0
        self.portfolio_size: int = 10

        self.initial_stock_price = 100.0
        self.expected_return = 0.0
        self.volatility = 0.2
        self.time_period = 1.0
        self.time_period_name = "Day"
        self.price_points: int = 1000
        self.time_step = 1.0 / self.price_points

        self.crossover_probability = 0.2
        self.mutation_probability = 0.01
        self.mutation_strength = 0.5
        self.tournament_size: int = 3

        self.values_lower_bound = 0.0
        self.values_upper_bound = 1.0

        self.mutation_deviation = 0.1

        self.num_top_scorers_shown: int = 10

        self.toolbox = base.Toolbox()
        self.pop = None
        self.hof = None
        self.stats = None

    def print_initial_stats(self):
        print(f'Initial stock price: ${repr(self.initial_stock_price)}')
        print(f'Expected return: {repr(self.expected_return * 100)}%')
        print(f'Volatility: {repr(self.volatility * 100)}%')
        print(f'Time period: {self.time_period} {self.time_period_name}{"s" if self.time_period != 1 else ""}')
        print(f'Time step: {repr(self.time_step)}')
        print(f'Price Points: {self.price_points}')
        print(f'Initial investment: ${repr(self.initial_investment)}')
        print(f'Number of stocks in portfolio: {self.portfolio_size}')
        print(f'Population size: {self.population_size}')
        print(f'Number of generations: {self.num_generations}')
        print(f'Crossover probability: {repr(self.crossover_probability * 100)}%')
        print(f'Mutation probability: {repr(self.mutation_probability * 100)}%')
        print(f'Mutation strength: {repr(self.mutation_strength * 100)}%')
        print(f'Mutation deviation: {repr(self.mutation_deviation * 100)}%')
        print(f'Tournament size: {self.tournament_size}')

    def generate_stock_data(self):
        stocks = [
            Stock(f'Stock{i}', self.initial_stock_price, self.expected_return, self.volatility, self.time_period, self.time_step,
                  random_seed=42 + i)
            for i in range(self.portfolio_size)]
        InvestorPortfolio.set_stocks(stocks)

    def setup_ga(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("InvestorPortfolio", InvestorPortfolio, fitness=creator.FitnessMax)
        self.toolbox.register("attr_sell", random.uniform, 0, 0.5)
        self.toolbox.register("attr_buy", random.uniform, 0, 0.5)
        self.toolbox.register("attr_stoploss", random.uniform, 0, 1.0)
        self.toolbox.register("attr_cash_ratio", random.uniform, 0, 1.0)

    def setup_population(self):
        def generate_new_investor():
            return creator.InvestorPortfolio(self.initial_investment,
                                             sell_threshold=self.toolbox.attr_sell(),
                                             buy_threshold=self.toolbox.attr_buy(),
                                             stop_loss_ratio=self.toolbox.attr_stoploss(),
                                             cash_ratio=self.toolbox.attr_cash_ratio())

        self.toolbox.register("investor", generate_new_investor)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.investor)
        self.toolbox.register("evaluate", evaluate_population)
        self.toolbox.register("mate", mate_investors)
        self.toolbox.register("mutate", mutate_investor, mu=0, sigma=self.mutation_deviation, indpb=self.mutation_probability,
                         lower_bound=self.values_lower_bound, upper_bound=self.values_upper_bound)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

    def generate_population(self):
        values = [
            (0.12329535956096988, 0.023, 0.4968684247982979, 0.9995108193746103),
            (0.12329535956096988, 0.023, 0.4968684247982979, 0.5),
            (0.2101871334368854, 0.1092842466932408, 0.5, 0.9345442264099624),
            (0.09066665291688599, 0.15307444708196606, 0.4549144125520966, 0.9997072061611406),
            (0.20472204319162064, 0.09918722982581912, 0.2183627036391925, 0.9805178638740347),
        ]

        top_performers = [
            creator.InvestorPortfolio(self.initial_investment, sell_threshold=a, buy_threshold=b, stop_loss_ratio=c,
                                      cash_ratio=d)
            for a, b, c, d in values
        ]

        self.pop = self.toolbox.population(n=self.population_size - len(top_performers)) + top_performers

    def setup_stats(self):
        self.hof = tools.HallOfFame(self.num_top_scorers_shown)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("median", np.median)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.stats.register("std", np.std)
        self.stats.register("var", np.var)

    def run_simulation(self):
        try:
            _, _ = custom_ea_simple(
                self.pop, self.toolbox,
                cxpb=self.crossover_probability,
                mutpb=self.mutation_probability,
                ngen=self.num_generations,
                stats=self.stats,
                halloffame=self.hof,
                verbose=True
            )
        except KeyboardInterrupt:
            print('Simulation interrupted by user')

    @staticmethod
    def print_individual(i, ind):
        print(f"Individual {i + 1} is:")
        print(ind.final_stats())

    def print_results(self):
        if self.hof:
            for i in range(0, len(self.hof)):
                individual = self.hof[i]
                self.print_individual(i, individual)

            print("Trade history of top performer:")

            for trade_history_item in self.hof[0].trade_history:
                print(trade_history_item)

            print("")

            print("Portfolio of top performer:")
            print(self.hof[0])

            print("")

            print("Other info of top performer:")
            self.print_individual(0, self.hof[0])

    def main(self):
        self.print_initial_stats()
        print('\n')
        print('Generating fake stock data')
        self.generate_stock_data()
        print('Setting up genetic algorithm library')
        self.setup_ga()
        print('Generating investor portfolios')
        self.setup_population()
        self.generate_population()
        self.setup_stats()
        print('Beginning simulation')
        self.run_simulation()
        self.print_results()
        print('Program complete')


class Stock:
    def __init__(self, name: str, initial_stock_price: float, expected_return: float, volatility: float,
                 time_period: float, time_step: float, random_seed: int = 42):
        self.name = name
        self.prices = np.array([], dtype="float32")
        self.parameters = (initial_stock_price, expected_return, volatility, time_period, time_step, random_seed)
        self.prices = self.calculate_geometric_brownian_motion(*self.parameters)

    @staticmethod
    def calculate_geometric_brownian_motion(initial_stock_price: float, expected_return: float, volatility: float,
                                            time_period: float, time_step: float, random_seed: int) -> np.ndarray:
        np.random.seed(random_seed)
        time_array = np.linspace(0, time_period, int(time_period / time_step), dtype="float32")
        num_steps = len(time_array)
        random_walk = np.random.standard_normal(size=num_steps)
        random_walk = np.cumsum(random_walk, dtype="float32") * np.sqrt(time_step)
        return np.array(initial_stock_price * np.exp(
            (expected_return - 0.5 * volatility ** 2) * time_array + volatility * random_walk), dtype="float32")


class InvestorPortfolio:
    stocks = []

    @classmethod
    def set_stocks(cls, stocks: List[Stock]):
        cls.stocks = stocks

    def __init__(self, initial_cash: float, sell_threshold: float, buy_threshold: float, stop_loss_ratio: float,
                 cash_ratio: float):
        self.initial_cash = initial_cash
        self.cash = self.initial_cash
        self.owned_stocks = {stock.name: 0 for stock in self.stocks}
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold
        self.stop_loss_ratio = stop_loss_ratio
        self.cash_ratio = cash_ratio
        self.target_cash = initial_cash
        self.num_buys = 0
        self.num_sells = 0
        self.trade_history = []

    def __eq__(self, other):
        same = self.sell_threshold == other.sell_threshold and \
               self.buy_threshold == other.buy_threshold and \
               self.stop_loss_ratio == other.stop_loss_ratio and \
               self.cash_ratio == other.cash_ratio

        return same

    def __str__(self):
        return self.string_representation(-1)

    def reset(self):
        self.cash = self.initial_cash
        for stock_name in self.owned_stocks:
            self.owned_stocks[stock_name] = 0

    def stocks_value(self, price_point):
        stock_values = [stock.prices[price_point] * self.owned_stocks[stock.name] for stock in self.stocks]
        return np.sum(stock_values)

    def portfolio_value(self, price_point):
        total_portfolio_value = self.cash + self.stocks_value(price_point)
        return total_portfolio_value

    def final_value(self):
        return self.portfolio_value(-1)

    def string_representation(self, price_point):
        string_representation = f"Cash: {self.cash}\n"
        string_representation += f"Stocks Value: {self.stocks_value(-1)}\n"
        for stock_name in self.owned_stocks:
            shares = self.owned_stocks[stock_name]
            price = None
            for stock in self.stocks:
                if stock.name == stock_name:
                    price = stock.prices[price_point]
                    break
            string_representation += f"{stock_name}: {shares} shares at ${price}/share, total ${shares * price}\n"
        return string_representation

    def final_stats(self):
        return f"\tInitial Cash ${repr(self.initial_cash)}\n" + \
              f"\tSell Threshold {repr(self.sell_threshold * 100)}%\n" + \
              f"\tBuy Threshold {repr(self.buy_threshold * 100)}%\n" + \
              f"\tStop Loss Ratio {repr(self.stop_loss_ratio * 100)}%\n" + \
              f"\tCash Ratio {repr(self.cash_ratio * 100)}%\n" + \
              f"\tFinal Cash: ${repr(self.final_value())}\n" + \
              f"\tFitness: {repr(self.fitness.values[0])}\n" + \
              f"\ttotal buys: {self.num_buys}\n" + \
              f"\ttotal sells: {self.num_sells}"

    def backtest_strategy(self, range_price_points, stock_prices, previous_buy_or_sell_prices):
        for price_point_num in range_price_points:
            current_prices = stock_prices[:, price_point_num]
            previous_prices = np.array([previous_buy_or_sell_prices[stock_name] for stock_name in self.owned_stocks])

            # Calculate the changes
            changes_from_previous_point = (current_prices - previous_prices) / previous_prices
            portfolio_value = self.cash + np.sum(stock_prices[:, price_point_num] * list(self.owned_stocks.values()))
            change_from_portfolio_value = (portfolio_value - self.target_cash) / self.target_cash
            is_stoploss = change_from_portfolio_value <= -self.stop_loss_ratio

            if portfolio_value > self.target_cash:
                self.target_cash = portfolio_value

            # Update the stocks and cash with a single loop
            for i, stock_name in enumerate(self.owned_stocks):
                n_stocks_original = self.owned_stocks[stock_name]
                current_price = current_prices[i]
                change_from_previous_point = changes_from_previous_point[i]

                if n_stocks_original > 0:
                    if change_from_previous_point >= self.sell_threshold or is_stoploss:
                        sell_price = current_price * n_stocks_original
                        self.cash += sell_price
                        self.owned_stocks[stock_name] = 0

                        previous_buy_or_sell_prices[stock_name] = current_price
                        self.num_sells += 1
                        if TRADE_HISTORY:
                            self.trade_history.append(f'{price_point_num}: Sold {n_stocks_original} share{"s" if n_stocks_original != 1 else ""} of {stock_name} at ${current_price}/share for ${sell_price}, cash ${self.cash}')
                else:
                    if change_from_previous_point <= -self.buy_threshold:
                        n_stocks = floor(self.cash * self.cash_ratio / current_price)
                        if n_stocks > 0:
                            buy_price = n_stocks * current_price
                            self.cash -= buy_price
                            self.owned_stocks[stock_name] = n_stocks
                            previous_buy_or_sell_prices[stock_name] = current_price
                            self.num_buys += 1
                            if TRADE_HISTORY:
                                self.trade_history.append(
                                    f'{price_point_num}: Bought {n_stocks} share{"s" if n_stocks != 1 else ""} of {stock_name} at ${current_price}/share for ${buy_price}, cash ${self.cash}')


def mate_investors(ind1: InvestorPortfolio, ind2: InvestorPortfolio, alpha: float = 0.5):
    ind1_attrs = [ind1.sell_threshold, ind1.buy_threshold, ind1.stop_loss_ratio, ind1.cash_ratio]
    ind2_attrs = [ind2.sell_threshold, ind2.buy_threshold, ind2.stop_loss_ratio, ind2.cash_ratio]

    offspring1_attrs, offspring2_attrs = tools.cxBlend(ind1_attrs, ind2_attrs, alpha)

    offspring1 = creator.InvestorPortfolio(ind1.initial_cash, *offspring1_attrs)
    offspring2 = creator.InvestorPortfolio(ind1.initial_cash, *offspring2_attrs)

    return offspring1, offspring2


def mutate_investor(individual: InvestorPortfolio, mu: float = 0, sigma: float = 0.1, indpb: float = 0.1, lower_bound: float = 0.0, upper_bound: float = 1.0):
    attrs = [individual.sell_threshold, individual.buy_threshold, individual.stop_loss_ratio, individual.cash_ratio]

    mutated_attrs, = tools.mutGaussian(attrs, mu=mu, sigma=sigma, indpb=indpb)

    # Ensure the values remain within the range [0, 1].
    mutated_attrs = [min(max(value, 0.0), 1.0) for value in mutated_attrs]

    individual.sell_threshold, individual.buy_threshold, individual.stop_loss_ratio, individual.cash_ratio = mutated_attrs

    return individual,


def preprocess_strategy_data(stocks):
    stock_names = [stock.name for stock in stocks]
    stock_prices = np.array([stock.prices for stock in stocks])
    return stock_names, stock_prices


def evaluate(investor_portfolio: InvestorPortfolio) -> Tuple[float]:
    investor_portfolio.reset()
    stock_names, stock_prices = preprocess_strategy_data(investor_portfolio.stocks)
    previous_buy_or_sell_prices = {stock_name: stock_prices[i, 0] for i, stock_name in enumerate(stock_names)}
    num_price_points = range(1, stock_prices.shape[1])
    investor_portfolio.backtest_strategy(num_price_points, stock_prices, previous_buy_or_sell_prices)
    return (investor_portfolio.final_value()/investor_portfolio.initial_cash) ** 4.0


def evaluate_population(population):
    results = [evaluate(individual) for individual in population]
    for ind, fit in zip(population, results):
        if np.isnan(fit):  # Check if fitness value is NaN
            ind.fitness.values = -np.inf,  # Set fitness to negative infinity if NaN
        else:
            ind.fitness.values = fit,


def pretty_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{seconds:.3f}s"


def custom_ea_simple(population, toolbox, cxpb, mutpb, ngen, stats=None,
                     halloffame=None, verbose=__debug__):
    generation_start_time = 0.0
    total_time = 0.0

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'generation_time', 'total_time', 'time_per_evaluation', 'estimated_total_time'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # Evaluate the entire population
    initial_evaluation_start_time = time.time()
    toolbox.evaluate(invalid_ind)
    initial_evaluation_time = time.time() - initial_evaluation_start_time
    total_time += initial_evaluation_time

    time_per_evaluation_initial = initial_evaluation_time / len(invalid_ind)

    estimated_total_time_initial = time_per_evaluation_initial * ngen * len(population)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind),
                   generation_time=pretty_time(initial_evaluation_time),
                   total_time=pretty_time(total_time),
                   time_per_evaluation=pretty_time(time_per_evaluation_initial),
                   estimated_total_time=pretty_time(estimated_total_time_initial),
                   **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # Evaluate the entire offspring population in parallel
        generation_start_time = time.time()
        toolbox.evaluate(invalid_ind)
        generation_time = time.time() - generation_start_time
        total_time += generation_time

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring[:-1] + [halloffame[0]]

        nevals = len(invalid_ind)

        # Calculate timing information
        if nevals != 0:
            time_per_evaluation = generation_time / nevals
        else:
            time_per_evaluation = generation_time
        estimated_total_time = total_time / gen * ngen

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record,
                       generation_time=pretty_time(generation_time),
                       total_time=pretty_time(total_time),
                       time_per_evaluation=pretty_time(time_per_evaluation),
                       estimated_total_time=pretty_time(estimated_total_time))
        if verbose:
            print(logbook.stream)

    return population, logbook


def main():
    simulator = StockTradingSimulation()
    simulator.main()


if __name__ == '__main__':
    profile = False
    if profile:
        import cProfile
        import pstats
        from io import StringIO

        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()
        # Print the profiling results.
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
    else:
        main()
