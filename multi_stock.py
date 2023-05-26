import asyncio

import numpy as np
from math import floor
from deap import base, creator, tools, algorithms
import random
from typing import List, Tuple
from scipy.stats import skew, kurtosis


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
        return initial_stock_price * np.exp(
            (expected_return - 0.5 * volatility ** 2) * time_array + volatility * random_walk)


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

    def reset(self):
        self.cash = self.initial_cash
        for stock_name in self.owned_stocks:
            self.owned_stocks[stock_name] = 0

    def final_value(self):
        stock_values = [stock.prices[-1] * self.owned_stocks[stock.name] for stock in self.stocks]
        total_portfolio_value = self.cash + sum(stock_values)
        return total_portfolio_value

    def backtest_strategy(self, range_price_points, stock_prices, previous_buy_or_sell_prices):
        stock_names = stock_prices.keys()
        for price_point_num in range_price_points:
            owned_stocks = {k: v for k, v in self.owned_stocks.items() if v > 0}
            for stock_name in stock_names:
                prices = stock_prices[stock_name]
                current_price = prices[price_point_num]
                previous_price = previous_buy_or_sell_prices[stock_name]

                if stock_name in owned_stocks:
                    change_from_previous_point = (current_price - previous_price) / previous_price
                    portfolio_value = self.cash + sum(
                        [stock.prices[price_point_num] * self.owned_stocks[stock.name] for stock in self.stocks])
                    change_from_portfolio_value = (portfolio_value - self.target_cash) / self.target_cash
                    is_stoploss = change_from_portfolio_value <= -self.stop_loss_ratio

                    if change_from_previous_point >= self.sell_threshold or is_stoploss:
                        self.cash += current_price * self.owned_stocks[stock_name]
                        if self.cash > self.target_cash:
                            self.target_cash = self.cash
                        self.owned_stocks[stock_name] = 0
                        previous_price = current_price
                        previous_buy_or_sell_prices[stock_name] = previous_price
                        self.num_sells += 1

                else:
                    change_from_previous_point = (current_price - previous_price) / previous_price
                    if change_from_previous_point <= -self.buy_threshold:
                        n_stocks = floor(self.cash * self.cash_ratio / current_price)
                        self.cash -= n_stocks * current_price
                        self.owned_stocks[stock_name] = n_stocks
                        previous_price = current_price
                        previous_buy_or_sell_prices[stock_name] = previous_price
                        self.num_buys += 1


def mate_investors(ind1: InvestorPortfolio, ind2: InvestorPortfolio):
    # Using a weighted average for mating
    weights = [random.random() for _ in range(4)]
    sum_weights = sum(weights)

    # Normalize the weights
    weights = [w / sum_weights for w in weights]

    # Create the offspring as a weighted average of parents' attributes
    ind1.sell_threshold = weights[0]*ind1.sell_threshold + weights[1]*ind2.sell_threshold
    ind2.sell_threshold = weights[2]*ind1.sell_threshold + weights[3]*ind2.sell_threshold

    ind1.buy_threshold = weights[0]*ind1.buy_threshold + weights[1]*ind2.buy_threshold
    ind2.buy_threshold = weights[2]*ind1.buy_threshold + weights[3]*ind2.buy_threshold

    ind1.stop_loss_ratio = weights[0]*ind1.stop_loss_ratio + weights[1]*ind2.stop_loss_ratio
    ind2.stop_loss_ratio = weights[2]*ind1.stop_loss_ratio + weights[3]*ind2.stop_loss_ratio

    ind1.cash_ratio = weights[0]*ind1.cash_ratio + weights[1]*ind2.cash_ratio
    ind2.cash_ratio = weights[2]*ind1.cash_ratio + weights[3]*ind2.cash_ratio

    return ind1, ind2


def mutate_investor(individual: InvestorPortfolio, mutation_probability: float, mutation_strength: float):
    # Ensure mutation_strength is within the expected range
    assert 0 <= mutation_strength <= 1, "mutation_strength should be between 0 and 1"

    # Calculate factor range based on mutation_strength
    lower_bound = 1 - mutation_strength * 0.5
    upper_bound = 1 + mutation_strength * 0.5

    # Using a multiplicative factor for mutation
    if random.random() < mutation_probability:
        factor = random.uniform(lower_bound, upper_bound)  # Factor randomly picked from [lower_bound, upper_bound]
        individual.sell_threshold = max(0, min(0.5, individual.sell_threshold * factor))
    if random.random() < mutation_probability:
        factor = random.uniform(lower_bound, upper_bound)
        individual.buy_threshold = max(0, min(0.5, individual.buy_threshold * factor))
    if random.random() < mutation_probability:
        factor = random.uniform(lower_bound, upper_bound)
        individual.stop_loss_ratio = max(0, min(0.5, individual.stop_loss_ratio * factor))
    if random.random() < mutation_probability:
        factor = random.uniform(lower_bound, upper_bound)
        individual.cash_ratio = max(0, min(1, individual.cash_ratio * factor))

    return individual,


async def evaluate(investor_portfolio: InvestorPortfolio) -> Tuple[float]:
    investor_portfolio.reset()
    stock_prices = {stock.name: stock.prices for stock in investor_portfolio.stocks}
    previous_buy_or_sell_prices = {stock_name: stock_price[0] for stock_name, stock_price in stock_prices.items()}
    num_price_points = range(1, len(investor_portfolio.stocks[0].prices))
    investor_portfolio.backtest_strategy(num_price_points, stock_prices, previous_buy_or_sell_prices)
    return investor_portfolio.final_value(),


async def evaluate_population_async(population):
    coroutines = [evaluate(individual) for individual in population]
    results = await asyncio.gather(*coroutines)
    return results


def evaluate_population(population):
    results = asyncio.run(evaluate_population_async(population))
    for ind, fit in zip(population, results):
        ind.fitness.values = fit


def custom_ea_simple(population, toolbox, cxpb, mutpb, ngen, stats=None,
                     halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # Evaluate the entire population in parallel
    toolbox.evaluate(invalid_ind)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
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
        toolbox.evaluate(invalid_ind)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def main():
    population_size = 50
    num_generations = 1000

    initial_investment = 1000.0
    portfolio_size = 20  # Number of stocks in the portfolio

    initial_stock_price = 100.0
    expected_return = 0.0
    volatility = 0.03
    time_period = 1.0
    time_step = 1.0 / 252.0 / 390.0  # Represents trading hours in a year
    price_points = int(round(time_period / time_step))  # might be one off

    crossover_probability = 0.6
    mutation_probability = 0.1
    mutation_strength = 0.9
    tournament_size = 3

    num_top_scorers_shown = 5

    print(f'Initial stock price: ${repr(initial_stock_price)}')
    print(f'Expected return: {repr(expected_return * 100)}%')
    print(f'Volatility: {repr(volatility * 100)}%')
    print(f'Time period: {time_period}')
    print(f'Time step: {repr(time_step)}')
    print(f'Price Points: {price_points}')
    print(f'Initial investment: ${repr(initial_investment)}')
    print(f'Number of stocks in portfolio: {portfolio_size}')
    print(f'Population size: {population_size}')
    print(f'Number of generations: {num_generations}')
    print(f'Crossover probability: {repr(crossover_probability * 100)}%')
    print(f'Mutation probability: {repr(mutation_probability * 100)}%')
    print(f'Mutation strength: {repr(mutation_strength * 100)}%')
    print(f'Tournament size: {tournament_size}')

    print('\n')

    print('Generating fake stock data')

    stocks = [
        Stock(f'Stock{i}', initial_stock_price, expected_return, volatility, time_period, time_step, random_seed=42 + i)
        for i in range(portfolio_size)]
    InvestorPortfolio.set_stocks(stocks)

    print('Setting up genetic algorithm library')

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("InvestorPortfolio", InvestorPortfolio, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_sell", random.uniform, 0, 0.5)
    toolbox.register("attr_buy", random.uniform, 0, 0.5)
    toolbox.register("attr_stoploss", random.uniform, 0, 0.5)
    toolbox.register("attr_cash_ratio", random.uniform, 0, 1.0)

    print('Generating investor portfolios')

    def generate_new_investor():
        return creator.InvestorPortfolio(initial_investment,
                                         sell_threshold=toolbox.attr_sell(),
                                         buy_threshold=toolbox.attr_buy(),
                                         stop_loss_ratio=toolbox.attr_stoploss(),
                                         cash_ratio=toolbox.attr_cash_ratio())

    toolbox.register("investor", generate_new_investor)

    toolbox.register("population", tools.initRepeat, list, toolbox.investor)

    toolbox.register("evaluate", evaluate_population)
    toolbox.register("mate", mate_investors)
    toolbox.register("mutate", mutate_investor, mutation_probability=mutation_probability, mutation_strength=mutation_strength)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    top_performers = [
        creator.InvestorPortfolio(initial_investment, sell_threshold=0.120, buy_threshold=0.023, stop_loss_ratio=0.500,
                                  cash_ratio=1.000),
        creator.InvestorPortfolio(initial_investment, sell_threshold=0.3, buy_threshold=0.1, stop_loss_ratio=0.2,
                                  cash_ratio=0.4),
        creator.InvestorPortfolio(initial_investment, sell_threshold=0.25, buy_threshold=0.15, stop_loss_ratio=0.18,
                                  cash_ratio=0.5),
    ]

    pop = toolbox.population(n=population_size - len(top_performers)) + top_performers

    hof = tools.HallOfFame(num_top_scorers_shown)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)
    stats.register("median", np.median)
    stats.register("var", np.var)
    stats.register("skewness", skew)
    stats.register("kurtosis", kurtosis)

    print('Beginning simulation')

    try:
        _, _ = custom_ea_simple(
            pop, toolbox,
            cxpb=crossover_probability,
            mutpb=mutation_probability,
            ngen=num_generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
    except KeyboardInterrupt:
        print('Simulation interrupted by user')
    except Exception as e:
        print(e)

    if hof:
        for i in range(0, len(hof)):
            individual: creator.InvestorPortfolio = hof[i]
            print(f"Individual {i + 1} is: \n"
                  f"\tInitial Cash ${repr(individual.initial_cash)}\n"
                  f"\tSell Threshold {repr(individual.sell_threshold * 100)}%\n"
                  f"\tBuy Threshold {repr(individual.buy_threshold * 100)}%\n"
                  f"\tStop Loss Ratio {repr(individual.stop_loss_ratio * 100)}%\n"
                  f"\tCash Ratio {repr(individual.cash_ratio * 100)}%\n"
                  f"\tFinal Cash: ${repr(individual.fitness.values[0])}\n"
                  f"\ttotal buys: {individual.num_buys}\n"
                  f"\ttotal sells: {individual.num_sells}\n")

    print('Program complete')


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
