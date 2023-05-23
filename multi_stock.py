import numpy as np
from math import floor
from deap import base, creator, tools, algorithms
import random
from typing import List, Tuple


class Stock:
    def __init__(self, name: str, S0: float, mu: float, sigma: float, T: float, dt: float):
        np.random.seed(42)
        self.name = name
        self.prices = None
        self.parameters = (S0, mu, sigma, T, dt)
        self.reset_prices()

    def reset_prices(self):
        self.prices = self.geometric_brownian_motion(*self.parameters)

    @staticmethod
    def geometric_brownian_motion(S0: float, mu: float, sigma: float, T: float, dt: float) -> np.ndarray:
        np.random.seed(42)
        t = np.linspace(0, T, int(T / dt))
        n = len(t)
        W = np.random.standard_normal(size=n)
        W = np.cumsum(W) * np.sqrt(dt)
        return S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)


class Portfolio:
    def __init__(self, stocks: List[Stock], initial_cash: float):
        self.stocks = {stock.name: stock for stock in stocks}
        self.initial_cash = initial_cash
        self.cash = self.initial_cash
        self.owned_stocks = {stock.name: 0 for stock in stocks}

    def reset(self):
        self.cash = self.initial_cash
        for stock_name in self.owned_stocks:
            self.owned_stocks[stock_name] = 0
        for stock in self.stocks:
            self.stocks[stock].reset_prices()

    def total_value(self):
        stock_values = [stock.prices[-1] * self.owned_stocks[stock.name] for stock in self.stocks.values()]
        total_portfolio_value = self.cash + sum(stock_values)
        return total_portfolio_value


class Investor:
    def __init__(self, portfolio: Portfolio, sell_threshold: float, buy_threshold: float, stop_loss_ratio: float, cash_ratio: float):
        self.portfolio = portfolio
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold
        self.stop_loss_ratio = stop_loss_ratio
        self.cash_ratio = cash_ratio
        self.target_cash = portfolio.initial_cash

    def backtest_strategy(self):
        self.portfolio.reset()
        stock_prices = {stock_name: stock.prices for stock_name, stock in self.portfolio.stocks.items()}
        previous_buy_or_sell_prices = {stock_name: stock_price[0] for stock_name, stock_price in stock_prices.items()}
        num_price_points = range(1, len(self.portfolio.stocks[next(iter(self.portfolio.stocks))].prices))
        self._buy_or_sell_stocks(num_price_points, stock_prices, previous_buy_or_sell_prices)

    def _buy_or_sell_stocks(self, num_price_points, stock_prices, previous_buy_or_sell_prices):
        for price_point_num in num_price_points:
            owned_stocks = {k: v for k, v in self.portfolio.owned_stocks.items() if v > 0}
            for stock_name, prices in stock_prices.items():
                current_price = prices[price_point_num]
                previous_price = previous_buy_or_sell_prices[stock_name]

                if stock_name in owned_stocks:
                    change_from_previous_point = (current_price - previous_price) / previous_price
                    portfolio_value = self.portfolio.cash + sum([stock.prices[price_point_num] * self.portfolio.owned_stocks[stock.name] for stock in self.portfolio.stocks.values()])
                    change_from_portfolio_value = (portfolio_value - self.target_cash) / self.target_cash
                    is_stoploss = change_from_portfolio_value <= -self.stop_loss_ratio

                    if change_from_previous_point >= self.sell_threshold or is_stoploss:
                        self.portfolio.cash += current_price * self.portfolio.owned_stocks[stock_name]
                        if self.portfolio.cash > self.target_cash:
                            self.target_cash = self.portfolio.cash
                        self.portfolio.owned_stocks[stock_name] = 0
                        previous_price = current_price
                        previous_buy_or_sell_prices[stock_name] = previous_price

                else:
                    change_from_previous_point = (current_price - previous_price) / previous_price
                    if change_from_previous_point <= -self.buy_threshold:
                        n_stocks = floor(self.portfolio.cash * self.cash_ratio / current_price)  # Considering cash_ratio
                        self.portfolio.cash -= n_stocks * current_price
                        self.portfolio.owned_stocks[stock_name] = n_stocks
                        previous_price = current_price
                        previous_buy_or_sell_prices[stock_name] = previous_price


def evaluate(investor: Investor) -> Tuple[float]:
    investor.backtest_strategy()
    return investor.portfolio.total_value(),


def mutate_investor(individual: Investor, mu: float, sigma: float, indpb: float):
    if random.random() < indpb:
        individual.sell_threshold = max(0, min(0.5, individual.sell_threshold + random.gauss(mu, sigma)))
    if random.random() < indpb:
        individual.buy_threshold = max(0, min(0.5, individual.buy_threshold + random.gauss(mu, sigma)))
    if random.random() < indpb:
        individual.stop_loss_ratio = max(0, min(0.5, individual.stop_loss_ratio + random.gauss(mu, sigma)))
    if random.random() < indpb:
        individual.cash_ratio = max(0, min(1, individual.cash_ratio + random.gauss(mu, sigma)))

    return individual,


def mate_investors(ind1: Investor, ind2: Investor):
    ind1.sell_threshold, ind2.sell_threshold = ind2.sell_threshold, ind1.sell_threshold
    ind1.buy_threshold, ind2.buy_threshold = ind2.buy_threshold, ind1.buy_threshold
    ind1.stop_loss_ratio, ind2.stop_loss_ratio = ind2.stop_loss_ratio, ind1.stop_loss_ratio
    ind1.cash_ratio, ind2.cash_ratio = ind2.cash_ratio, ind1.cash_ratio

    return ind1, ind2


def main():
    population = 100
    generations = 15

    initial_cash = 10000
    num_stocks = 5  # Number of stocks in the portfolio

    S0: float = 100
    mu: float = 0
    sigma: float = 0.2
    T: float = 1
    dt: float = 1 / 252 / 390

    print(f'S0 {S0}, mu {mu}, sigma {sigma}, T {T}, dt {dt}, initial cash {initial_cash}, num stocks {num_stocks}')

    stocks = [Stock(f'Stock{i}', S0, mu, sigma, T, dt) for i in range(num_stocks)]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Investor", Investor, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_sell", random.uniform, 0, 0.5)
    toolbox.register("attr_buy", random.uniform, 0, 0.5)
    toolbox.register("attr_stoploss", random.uniform, 0, 0.5)
    toolbox.register("attr_cash_ratio", random.uniform, 0, 1.0)

    def generate_new_investor():
        return creator.Investor(Portfolio(stocks, initial_cash),
                                sell_threshold=toolbox.attr_sell(),
                                buy_threshold=toolbox.attr_buy(),
                                stop_loss_ratio=toolbox.attr_stoploss(),
                                cash_ratio=toolbox.attr_cash_ratio())  # Include cash_ratio

    toolbox.register("investor", generate_new_investor)

    toolbox.register("population", tools.initRepeat, list, toolbox.investor)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", mate_investors)
    toolbox.register("mutate", mutate_investor, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats, halloffame=hof, verbose=True)
    except KeyboardInterrupt:
        pass

    if hof:
        for i in range(0, len(hof)):
            individual = hof[i]
            print(f"Individual {i+1} is: sell_threshold={individual.sell_threshold:.3f}, "
                  f"buy_threshold={individual.buy_threshold:.3f}, "
                  f"stop_loss_ratio={individual.stop_loss_ratio:.3f}, "
                  f"cash_ratio={hof[i].cash_ratio:.3f}\n"
                  f"with fitness: {individual.fitness}")


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
