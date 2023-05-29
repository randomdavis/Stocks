import deap.tools
import numpy as np
from math import floor
from deap import base, creator, tools, algorithms
import random

from typing import List
import time

import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_stock_prices(stocks):
    fig = make_subplots(rows=len(stocks), cols=1, shared_xaxes=True, subplot_titles=[stock.name for stock in stocks])
    for i, stock in enumerate(stocks, start=1):
        fig.add_trace(go.Scatter(y=stock.prices, x=list(range(len(stock.prices))),
                                 mode='lines', name=stock.name), row=i, col=1)
    fig.update_layout(height=200*len(stocks), width=900, title_text="Stock Prices over Time")
    fig.show()


def plot_signals(best_individual):
    fig = make_subplots(rows=len(best_individual.stocks), cols=1, shared_xaxes=True,
                        subplot_titles=[stock.name for stock in best_individual.stocks])
    for i, stock in enumerate(best_individual.stocks, start=1):
        fig.add_trace(go.Scatter(y=stock.prices, x=list(range(len(stock.prices))),
                                 mode='lines', name=stock.name), row=i, col=1)
        buys = [trade for trade in best_individual.trade_history if 'Buy' in trade and stock.name in trade]
        sells = [trade for trade in best_individual.trade_history if 'Sell' in trade and stock.name in trade]
        buy_x, buy_y = [], []
        for buy in buys:
            idx = int(buy.split(': ')[0])
            buy_x.append(idx)
            buy_y.append(stock.prices[idx])
        sell_x, sell_y = [], []
        for sell in sells:
            idx = int(sell.split(': ')[0])
            sell_x.append(idx)
            sell_y.append(stock.prices[idx])
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers',
                                 marker=dict(size=10, color='green', symbol='triangle-up'), name=f'Buy {stock.name}'),
                      row=i, col=1)
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers',
                                 marker=dict(size=10, color='red', symbol='triangle-down'), name=f'Sell {stock.name}'),
                      row=i, col=1)
    fig.update_layout(height=200 * len(best_individual.stocks), width=900,
                      title_text="Best Individual's Buy/Sell signals")
    fig.show()


def plot_fitness_evolution(logbook):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=logbook.select('gen'), y=logbook.select('avg'),
                             mode='lines', name='Average'))
    fig.add_trace(go.Scatter(x=logbook.select('gen'), y=logbook.select('max'),
                             mode='lines', name='Maximum'))
    fig.add_trace(go.Scatter(x=logbook.select('gen'), y=logbook.select('min'),
                             mode='lines', name='Minimum'))
    fig.update_layout(title='Evolution of Fitness over Generations',
                      xaxis_title='Generation', yaxis_title='Fitness')
    fig.show()


def plot_histogram(population, title):
    fitness_values = [ind.fitness.values[0] for ind in population]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=fitness_values, nbinsx=20))
    fig.update_layout(title=title,
                      xaxis_title='Fitness', yaxis_title='Count')
    fig.show()


random.seed(42)
np.random.seed(42)

TRADE_HISTORY = True


class StockTradingSimulation:
    def __init__(self):
        self.population_size: int = 500
        self.num_generations: int = 10000
        self.initial_investment = 10000.0
        self.portfolio_size: int = 10
        self.initial_stock_price = 300.0
        self.expected_return = -0.5
        self.volatility = 0.2
        self.time_period = 1.0
        self.time_period_name = "Day"
        self.price_points: int = 1000
        self.time_step = 1.0 / self.price_points
        self.crossover_probability = 0.6
        self.mutation_probability = 0.1
        self.mutation_strength = 0.8
        self.tournament_size: int = 7
        self.values_lower_bound = 0.0
        self.values_upper_bound = 1.0
        self.mutation_deviation = 0.1
        self.num_top_scorers_shown: int = 10
        self.toolbox = base.Toolbox()
        self.pop = None
        self.hof = None
        self.stats = None
        self.best_individual = None
        self.logbook = None
        self.populations_to_store = {0: None, self.num_generations // 2: None, self.num_generations: None}

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
        self.toolbox.register("attr_buy_ratio", random.uniform, 0, 1.0)
        self.toolbox.register("attr_sell_ratio", random.uniform, 0, 1.0)

    def setup_population(self):
        def generate_new_investor():
            return creator.InvestorPortfolio(self.initial_investment,
                                             sell_threshold=self.toolbox.attr_sell(),
                                             buy_threshold=self.toolbox.attr_buy(),
                                             stop_loss_ratio=self.toolbox.attr_stoploss(),
                                             buy_ratio=self.toolbox.attr_buy_ratio(),
                                             sell_ratio=self.toolbox.attr_sell_ratio())

        self.toolbox.register("investor", generate_new_investor)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.investor)
        self.toolbox.register("evaluate", evaluate_population)
        self.toolbox.register("mate", mate_investors, alpha=0.5, lower_bound=self.values_lower_bound, upper_bound=self.values_upper_bound)
        self.toolbox.register("mutate", mutate_investor, mu=0, sigma=self.mutation_deviation, indpb=self.mutation_probability,
                         lower_bound=self.values_lower_bound, upper_bound=self.values_upper_bound)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

    def generate_population(self):
        values = [
            (0.12329535956096988, 0.023, 0.4968684247982979, 0.9995108193746103, 1.0),
            (0.12329535956096988, 0.023, 0.4968684247982979, 0.5, 1.0),
            (0.2101871334368854, 0.1092842466932408, 0.5, 0.9345442264099624, 1.0),
            (0.09066665291688599, 0.15307444708196606, 0.4549144125520966, 0.9997072061611406, 1.0),
            (0.20230756367917923, 0.1088805122404803, 0.9893778894122727, 0.9939994206565086, 1.0),
        ]

        top_performers = [
            creator.InvestorPortfolio(self.initial_investment, sell_threshold=a, buy_threshold=b, stop_loss_ratio=c,
                                      buy_ratio=d, sell_ratio=e)
            for a, b, c, d, e in values
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
            self.custom_ea_simple(verbose=True)
        except KeyboardInterrupt:
            print('Simulation interrupted by user')
        if self.hof:
            self.best_individual = self.hof[0]
            self.populations_to_store[self.num_generations] = self.pop.copy()
            self.populations_to_store[self.num_generations // 2] = self.pop.copy()

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

    def custom_ea_simple(self, verbose=__debug__):
        total_time = 0.0
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals', 'generation_time', 'total_time', 'time_per_evaluation',
                          'estimated_total_time'] + (self.stats.fields if self.stats else [])
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        # Evaluate the entire population
        initial_evaluation_start_time = time.time()
        self.toolbox.evaluate(invalid_ind)
        initial_evaluation_time = time.time() - initial_evaluation_start_time
        total_time += initial_evaluation_time
        time_per_evaluation_initial = initial_evaluation_time / len(invalid_ind)
        estimated_total_time_initial = time_per_evaluation_initial * self.num_generations * len(self.pop)
        if self.hof is not None:
            self.hof.update(self.pop)
        record = self.stats.compile(self.pop) if self.stats else {}
        self.logbook.record(gen=0, nevals=len(invalid_ind),
                       generation_time=pretty_time(initial_evaluation_time),
                       total_time=pretty_time(total_time),
                       time_per_evaluation=pretty_time(time_per_evaluation_initial),
                       estimated_total_time=pretty_time(estimated_total_time_initial),
                       **record)
        if verbose:
            print(self.logbook.stream)

        # Begin the generational process
        for gen in range(1, self.num_generations + 1):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))
            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, self.toolbox, self.crossover_probability, self.mutation_probability)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # Evaluate the entire offspring population in parallel
            generation_start_time = time.time()
            self.toolbox.evaluate(invalid_ind)
            generation_time = time.time() - generation_start_time
            total_time += generation_time
            # Update the hall of fame with the generated individuals
            if self.hof is not None:
                self.hof.update(offspring)
            # Replace the current population by the offspring
            self.pop[:] = offspring[:-1] + [self.hof[0]]
            # Store the population if the current generation is in populations_to_store
            if gen - 1 in self.populations_to_store:
                self.populations_to_store[gen - 1] = self.pop.copy()
            nevals = len(invalid_ind)
            # Calculate timing information
            if nevals != 0:
                time_per_evaluation = generation_time / nevals
            else:
                time_per_evaluation = generation_time
            estimated_total_time = total_time / gen * self.num_generations
            # Append the current generation statistics to the logbook
            record = self.stats.compile(self.pop) if self.stats else {}
            self.logbook.record(gen=gen, nevals=nevals, **record,
                           generation_time=pretty_time(generation_time),
                           total_time=pretty_time(total_time),
                           time_per_evaluation=pretty_time(time_per_evaluation),
                           estimated_total_time=pretty_time(estimated_total_time))
            if verbose:
                print(self.logbook.stream)


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
                 buy_ratio: float, sell_ratio: float):
        self.initial_cash = initial_cash
        self.cash = self.initial_cash
        self.owned_stocks = {stock.name: 0 for stock in self.stocks}
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold
        self.stop_loss_ratio = stop_loss_ratio
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        self.target_cash = initial_cash
        self.num_buys = 0
        self.num_sells = 0
        self.trade_history = []
        self.previous_buy_or_sell_prices = {}

    def __eq__(self, other):
        same = self.sell_threshold == other.sell_threshold and \
               self.buy_threshold == other.buy_threshold and \
               self.stop_loss_ratio == other.stop_loss_ratio and \
               self.buy_ratio == other.buy_ratio and \
               self.sell_ratio == other.sell_ratio

        return same

    def __str__(self):
        return self.string_representation(-1)

    def reset(self):
        self.cash = self.initial_cash
        for stock_name in self.owned_stocks:
            self.owned_stocks[stock_name] = 0

    def stocks_value(self, price_point):
        total_stock_value = 0.0
        for stock in self.stocks:
            total_stock_value += stock.prices[price_point] * self.owned_stocks[stock.name]
        return total_stock_value

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
              f"\tBuy Ratio {repr(self.buy_ratio * 100)}%\n" + \
              f"\tSell Ratio {repr(self.sell_ratio * 100)}%\n" + \
              f"\tFinal Cash: ${repr(self.final_value())}\n" + \
              f"\tFitness: {repr(self.fitness.values[0])}\n" + \
              f"\ttotal buys: {self.num_buys}\n" + \
              f"\ttotal sells: {self.num_sells}"

    def update_cash_stocks_owned(self, operation, stock_name, current_price, n_stocks, price_point_num):
        if n_stocks > 0:
            total_stock_value = current_price * n_stocks
            if operation == "buy":
                self.cash -= total_stock_value
                self.owned_stocks[stock_name] += n_stocks
                self.num_buys += 1
            elif operation == "sell":
                self.cash += total_stock_value
                self.owned_stocks[stock_name] -= n_stocks
                self.num_sells += 1
            assert self.cash >= 0.0
            if TRADE_HISTORY:
                self.trade_history.append(f'{price_point_num}: {operation.title()} {n_stocks} share{"s" if n_stocks != 1 else ""} of {stock_name} at ${current_price}/share for ${round(current_price * n_stocks, 2)}, cash ${round(self.cash, 2)}')
            self.previous_buy_or_sell_prices[stock_name] = current_price

    def sell_condition_met(self, stock_name, current_price, portfolio_val):
        change_from_previous_point = (current_price - self.previous_buy_or_sell_prices[stock_name]) / self.previous_buy_or_sell_prices[stock_name]
        return change_from_previous_point >= self.sell_threshold or portfolio_val <= (1 - self.stop_loss_ratio) * self.target_cash

    def buy_condition_met(self, stock_name, current_price):
        change_from_previous_point = (current_price - self.previous_buy_or_sell_prices[stock_name]) / self.previous_buy_or_sell_prices[stock_name]
        return change_from_previous_point <= -self.buy_threshold

    def execute_decision(self, stock_name, current_price, price_point_num):
        n_stocks_original = self.owned_stocks[stock_name]
        if n_stocks_original > 0 and self.sell_condition_met(stock_name, current_price, price_point_num):
            n_stocks = floor(self.sell_ratio * n_stocks_original)
            if n_stocks > 0:
                self.update_cash_stocks_owned("sell", stock_name, current_price, n_stocks, price_point_num)
        elif self.buy_condition_met(stock_name, current_price):
            n_stocks = floor(self.cash * self.buy_ratio / current_price)
            if n_stocks > 0:
                self.update_cash_stocks_owned("buy", stock_name, current_price, n_stocks, price_point_num)

    def backtest_strategy(self):
        self.previous_buy_or_sell_prices = {stock.name: stock.prices[0] for stock in self.stocks}
        stock_prices_dict = {stock.name: stock.prices for stock in self.stocks}
        num_price_points = len(self.stocks[0].prices)
        range_price_points = range(num_price_points)
        for price_point_num in range_price_points:
            portfolio_val = self.portfolio_value(price_point_num)
            for stock_name in self.owned_stocks:
                current_price = stock_prices_dict[stock_name][price_point_num]
                self.execute_decision(stock_name, current_price, price_point_num)
            if portfolio_val > self.target_cash:
                self.target_cash = portfolio_val


def mate_investors(ind1: InvestorPortfolio, ind2: InvestorPortfolio, alpha: float = 0.5, lower_bound=0.0, upper_bound=1.0):
    ind1_attrs = [ind1.sell_threshold, ind1.buy_threshold, ind1.stop_loss_ratio, ind1.buy_ratio, ind1.sell_ratio]
    ind2_attrs = [ind2.sell_threshold, ind2.buy_threshold, ind2.stop_loss_ratio, ind2.buy_ratio, ind2.sell_ratio]
    offspring1_attrs, offspring2_attrs = tools.cxBlend(ind1_attrs, ind2_attrs, alpha)
    offspring1_attrs = [min(max(value, 0.0), 1.0) for value in offspring1_attrs]
    offspring2_attrs = [min(max(value, 0.0), 1.0) for value in offspring2_attrs]
    offspring1 = creator.InvestorPortfolio(ind1.initial_cash, *offspring1_attrs)
    offspring2 = creator.InvestorPortfolio(ind1.initial_cash, *offspring2_attrs)
    return offspring1, offspring2


def mutate_investor(individual: InvestorPortfolio, mu: float = 0, sigma: float = 0.1, indpb: float = 0.1, lower_bound: float = 0.0, upper_bound: float = 1.0):
    attrs = [individual.sell_threshold, individual.buy_threshold, individual.stop_loss_ratio, individual.buy_ratio, individual.sell_ratio]
    mutated_attrs, = tools.mutGaussian(attrs, mu=mu, sigma=sigma, indpb=indpb)
    mutated_attrs = [min(max(value, lower_bound), upper_bound) for value in mutated_attrs]
    individual.sell_threshold, individual.buy_threshold, individual.stop_loss_ratio, individual.buy_ratio, individual.sell_ratio = mutated_attrs
    return individual,


def preprocess_strategy_data(stocks):
    stock_names = [stock.name for stock in stocks]
    stock_prices = np.array([stock.prices for stock in stocks])
    return stock_names, stock_prices


def evaluate(investor_portfolio: InvestorPortfolio):
    investor_portfolio.reset()
    investor_portfolio.backtest_strategy()
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


def main():
    simulator = StockTradingSimulation()
    simulator.main()
    if InvestorPortfolio.stocks:
        plot_stock_prices(InvestorPortfolio.stocks)
    if simulator.best_individual:
        plot_signals(simulator.best_individual)
    if simulator.logbook:
        plot_fitness_evolution(simulator.logbook)
        plot_histogram(simulator.populations_to_store[0], 'Histogram of Initial Population Fitness')
        plot_histogram(simulator.populations_to_store[simulator.num_generations // 2], 'Histogram of Mid-point Population Fitness')
        plot_histogram(simulator.populations_to_store[simulator.num_generations], 'Histogram of Final Population Fitness')


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
