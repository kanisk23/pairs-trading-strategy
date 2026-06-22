import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


#data import
data = yf.download(["JPM","BAC","C","WFC","GS","MS","HSBC","UBS","SAN","BBVA"], start = "2015-01-01", end = "2019-01-01")["Close"]
data = data.dropna()
train = data.loc["2016-01-01":"2016-12-31"]
test  = data.loc["2017-01-01":"2018-12-31"]
returns = np.log(train/train.shift(1))
returns = returns.dropna()
#correlation test
corr_matrix = returns.corr()
corr_matrix.index.name = None
corr_matrix.columns.name = None
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
pairs = upper.stack().reset_index()
pairs.columns = ['stock1', 'stock2', 'correlation']
pairs=pairs.dropna()
valid_pairs=pairs[pairs["correlation"]>0.7]

#cointegration test
results = []
for _, row in valid_pairs.iterrows():
    stock1 = row['stock1']
    stock2 = row['stock2']
    s1 = train[stock1]
    s2 = train[stock2]
    score, pvalue, critical_values = coint(s1, s2)
    results.append({
        'stock1': stock1,
        'stock2': stock2,
        'correlation': row['correlation'],
        'coint_score': score,
        'pvalue': pvalue
    })

coint_results = pd.DataFrame(results)
coint_results = coint_results.sort_values('pvalue')
cointegrated_pairs = coint_results[coint_results["pvalue"] < 0.05]
if len(cointegrated_pairs) == 0:
    raise ValueError("No cointegrated pairs found.")


# loop over ALL cointegrated pairs
all_metrics = []
all_cum_returns = {}

for _, best_pair in cointegrated_pairs.iterrows():

    #hedge ratio
    stock1 = best_pair["stock1"]
    stock2 = best_pair["stock2"]
    y = train[stock1]
    x = sm.add_constant(train[stock2])
    model = sm.OLS(y, x).fit()
    hedge_ratio = model.params[stock2]

    #spread

    spread = test[stock1] - hedge_ratio * test[stock2]


    train_spread = train[stock1] - hedge_ratio * train[stock2]

    spread_mean = train_spread.mean()
    spread_std = train_spread.std()
    adf_stat, pvalue, *_ = adfuller(train_spread)

    print(f"ADF p-value      : {pvalue:.4f}")

    zscore = (spread - spread_mean) / spread_std
    #trading signals
    signals = pd.DataFrame(index=zscore.index)
    signals["signal"] = np.nan
    signals["zscore"] = zscore
    signals["signal"] = np.nan
    signals.loc[zscore > 2, "signal"] = -1
    signals.loc[zscore < -2, "signal"] = 1
    signals.loc[abs(zscore) < 0.5, "signal"] = 0
    signals["signal"] = signals["signal"].ffill()
    signals["signal"] = signals["signal"].fillna(0)

    signals["stock1_position"] = signals["signal"]
    signals["stock2_position"] = -signals["signal"] * hedge_ratio
    returns1 = np.log(test[stock1] / test[stock1].shift(1))
    returns2 = np.log(test[stock2] / test[stock2].shift(1))

    strategy_returns = (
        signals["stock1_position"].shift(1) * returns1 +
        signals["stock2_position"].shift(1) * returns2
    )
    strategy_returns = strategy_returns.dropna()
    cum_returns = np.exp(strategy_returns.cumsum())

    # Performance Metrics

    total_return = cum_returns.iloc[-1] - 1

    annual_return = (
        cum_returns.iloc[-1] ** (252 / len(cum_returns))
    ) - 1

    volatility = strategy_returns.std() * np.sqrt(252)

    sharpe = annual_return / volatility if volatility != 0 else np.nan

    # Maximum Drawdown
    rolling_max = cum_returns.cummax()
    drawdown = cum_returns / rolling_max - 1
    max_drawdown = drawdown.min()

    # Average Daily Return
    avg_daily_return = strategy_returns.mean()

    # Number of Trades
    entries = (
        ((signals["signal"] == 1) &
         (signals["signal"].shift(1) != 1))
        |
        ((signals["signal"] == -1) &
         (signals["signal"].shift(1) != -1))
    )
    num_trades = entries.sum()
    # sortino
    downside = strategy_returns[strategy_returns < 0]

    downside_vol = downside.std() * np.sqrt(252)

    sortino = (
        annual_return / downside_vol
        if downside_vol != 0
        else np.nan
    )
    # Win Rate
    winning_days = (strategy_returns > 0).sum()
    total_days = (strategy_returns != 0).sum()

    if total_days > 0:
        win_rate = winning_days / total_days
    else:
        win_rate = np.nan

    print("\n===== PERFORMANCE REPORT =====")
    print(f"Pair: {stock1} / {stock2}")
    print(f"Hedge Ratio: {hedge_ratio:.4f}")
    print()
    print(f"Total Return      : {total_return:.2%}")
    print(f"Annual Return     : {annual_return:.2%}")
    print(f"Volatility        : {volatility:.2%}")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Max Drawdown      : {max_drawdown:.2%}")
    print(f"Avg Daily Return  : {avg_daily_return:.4%}")
    print(f"Number of Trades  : {num_trades}")
    print(f"Win Rate          : {win_rate:.2%}")
    print(f"Sortino           : {sortino:.2%}")
    #plotting
    long_entries = signals["signal"].diff() == 1
    short_entries = signals["signal"].diff() == -1

    fig, ax = plot.subplots(3, 1, figsize=(12, 10))

    # Spread

    ax[0].plot(spread)
    ax[0].set_title("Spread")

    ax[0].grid(True)
    ax[0].axhline(spread_mean, linestyle="--")
    ax[0].axhline(spread_mean + 2*spread_std, linestyle=":")
    ax[0].axhline(spread_mean - 2*spread_std, linestyle=":")

    # Z-score
    ax[1].plot(zscore)

    ax[1].scatter(
        zscore.index[long_entries],
        zscore[long_entries],
        marker="^",
        s=80,
        label="Long Entry"
    )

    ax[1].scatter(
        zscore.index[short_entries],
        zscore[short_entries],
        marker="v",
        s=80,
        label="Short Entry"
    )

    ax[1].axhline(2, linestyle="--")
    ax[1].axhline(-2, linestyle="--")
    ax[1].axhline(0)
    ax[1].legend()
    ax[1].set_title("Z-Score Signals")
    ax[1].grid(True)

    # Equity Curve
    ax[2].plot(cum_returns)
    ax[2].set_title("Strategy Equity Curve")
    ax[2].grid(True)

    plot.tight_layout()
    plot.show()

    # store for comparison
    all_metrics.append({
        "pair": f"{stock1}/{stock2}",
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_daily_return": avg_daily_return,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "sortino": sortino,
    })
    all_cum_returns[f"{stock1}/{stock2}"] = cum_returns


# comparison table 
print("\n===== PAIR COMPARISON TABLE (sorted by Sharpe) =====")
comparison = pd.DataFrame(all_metrics).sort_values("sharpe", ascending=False)
comparison = comparison.set_index("pair")
print(comparison.to_string())

# overlaid equity curves 
fig, ax = plot.subplots(figsize=(12, 5))
for pair_name, cr in all_cum_returns.items():
    ax.plot(cr, label=pair_name)
ax.axhline(1, linestyle="--", color="black")
ax.set_title("Equity Curves — All Cointegrated Pairs")
ax.legend(fontsize=8)
ax.grid(True)
plot.tight_layout()
plot.show()