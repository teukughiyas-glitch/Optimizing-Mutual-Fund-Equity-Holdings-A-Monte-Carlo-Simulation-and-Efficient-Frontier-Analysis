from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "2021-01-01"
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05
MAX_WEIGHT = 0.1394
NUM_SIMULATIONS = 10_000
RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "portfolio_outputs"

observed_weights = {
	"ASII.JK": 0.0758,
	"BBCA.JK": 0.0849,
	"BMRI.JK": 0.1234,
	"BBNI.JK": 0.0990,
	"BBRI.JK": 0.1309,
	"BBTN.JK": 0.0962,
	"BUMI.JK": 0.1034,
	"INKP.JK": 0.0907,
	"PGAS.JK": 0.0839,
	"TINS.JK": 0.1118,
}

TICKERS = list(observed_weights)


if Path(__file__).resolve().parent.name == "portfolio_outputs":
	OUTPUT_DIR = Path(__file__).resolve().parent


def download_adjusted_prices():
	"""Download total-return prices; Yahoo's Adj Close includes cash dividends."""
	end_date = date.today() + timedelta(days=1)
	prices = yf.download(
		TICKERS,
		start=START_DATE,
		end=end_date.isoformat(),
		auto_adjust=False,
		progress=False,
		group_by="column",
	)

	if prices.empty:
		raise RuntimeError("Data harga tidak berhasil diunduh dari Yahoo Finance.")

	if isinstance(prices.columns, pd.MultiIndex):
		if "Adj Close" not in prices.columns.get_level_values(0):
			raise RuntimeError("Kolom Adj Close tidak tersedia untuk data saham.")
		prices = prices["Adj Close"]
	elif "Adj Close" in prices.columns:
		prices = prices[["Adj Close"]]

	missing = [ticker for ticker in TICKERS if ticker not in prices.columns]
	if missing:
		raise RuntimeError(f"Data Adj Close tidak tersedia untuk: {', '.join(missing)}")

	prices = prices[TICKERS].dropna(how="all").ffill().dropna()
	if prices.empty:
		raise RuntimeError("Tidak ada data harga lengkap setelah pembersihan.")
	return prices


def normalize_weights(weights):
	total = sum(weights.values())
	if total <= 0:
		raise ValueError("Total bobot harus lebih besar dari nol.")
	normalized = pd.Series(weights, dtype=float).reindex(TICKERS) / total
	if normalized.isna().any():
		raise ValueError("Semua ticker harus memiliki bobot.")
	return normalized


def portfolio_metrics(weights, mean_returns, covariance):
	annual_return = float(weights @ mean_returns * TRADING_DAYS_PER_YEAR)
	annual_variance = float(weights @ (covariance * TRADING_DAYS_PER_YEAR) @ weights)
	annual_volatility = float(np.sqrt(max(annual_variance, 0.0)))
	return annual_return, annual_volatility


def sharpe_ratio(annual_return, annual_volatility):
	if annual_volatility == 0:
		return np.nan
	return (annual_return - RISK_FREE_RATE) / annual_volatility


def generate_weights(num_portfolios, num_assets, max_weight, seed):
	"""Generate fully invested long-only portfolios under the per-stock cap."""
	rng = np.random.default_rng(seed)
	accepted = []
	while len(accepted) < num_portfolios:
		candidates = rng.dirichlet(np.ones(num_assets), size=num_portfolios)
		accepted.extend(candidate for candidate in candidates if candidate.max() <= max_weight)
	return np.asarray(accepted[:num_portfolios])


def main():
	OUTPUT_DIR.mkdir(exist_ok=True)
	prices = download_adjusted_prices()
	daily_returns = prices.pct_change().dropna()
	mean_returns = daily_returns.mean().to_numpy(dtype=float)
	covariance = daily_returns.cov().to_numpy(dtype=float)

	fund_weights = normalize_weights(observed_weights).to_numpy(dtype=float)
	if fund_weights.max() > MAX_WEIGHT:
		raise ValueError("Ada bobot mutual fund yang melebihi batas 13,94%.")

	simulated_weights = generate_weights(
		NUM_SIMULATIONS, len(TICKERS), MAX_WEIGHT, RANDOM_SEED
	)
	metrics = np.array(
		[portfolio_metrics(weights, mean_returns, covariance) for weights in simulated_weights]
	)
	simulation = pd.DataFrame(
		{
			"Simulation": np.arange(1, NUM_SIMULATIONS + 1),
			"Annualized Return": metrics[:, 0],
			"Annualized Volatility": metrics[:, 1],
		}
	)
	simulation["Sharpe Ratio (rf=5%)"] = (
		simulation["Annualized Return"] - RISK_FREE_RATE
	) / simulation["Annualized Volatility"]
	simulation = pd.concat(
		[simulation, pd.DataFrame(simulated_weights, columns=TICKERS)], axis=1
	)
	best_simulation = simulation.loc[simulation["Sharpe Ratio (rf=5%)"].idxmax()]

	fund_return, fund_volatility = portfolio_metrics(fund_weights, mean_returns, covariance)
	fund_sharpe = sharpe_ratio(fund_return, fund_volatility)
	fund_row = pd.DataFrame(
		[{
			"Portfolio": "Mutual fund observed",
			"Annualized Return": fund_return,
			"Annualized Volatility": fund_volatility,
			"Sharpe Ratio (rf=5%)": fund_sharpe,
		}]
	)
	comparison = pd.concat(
		[
			fund_row,
			pd.DataFrame(
				[{
					"Portfolio": "Best simulation Sharpe",
					"Annualized Return": best_simulation["Annualized Return"],
					"Annualized Volatility": best_simulation["Annualized Volatility"],
					"Sharpe Ratio (rf=5%)": best_simulation["Sharpe Ratio (rf=5%)"],
				}]
			),
		],
		ignore_index=True,
	)

	prices.to_csv(OUTPUT_DIR / "mutual_fund_adjusted_prices.csv")
	simulation.to_csv(OUTPUT_DIR / "mutual_fund_10000_simulations.csv", index=False)
	comparison.to_csv(OUTPUT_DIR / "mutual_fund_comparison.csv", index=False)
	best_composition = pd.DataFrame({
		"Ticker": TICKERS,
		"Best Portfolio Weight": best_simulation[TICKERS].to_numpy(dtype=float),
	})
	best_composition["Best Portfolio Weight (%)"] = best_composition["Best Portfolio Weight"] * 100
	best_composition.to_csv(OUTPUT_DIR / "best_portfolio_composition.csv", index=False)
	allocation_comparison = pd.DataFrame({
		"Ticker": TICKERS,
		"Observed Top-10 Weight (%)": fund_weights * 100,
		"Maximum Sharpe Weight (%)": best_simulation[TICKERS].to_numpy(dtype=float) * 100,
	})
	allocation_comparison.to_csv(OUTPUT_DIR / "observed_vs_maximum_sharpe_weights.csv", index=False)

	plt.figure(figsize=(11, 7))
	scatter = plt.scatter(
		simulation["Annualized Volatility"] * 100,
		simulation["Annualized Return"] * 100,
		c=simulation["Sharpe Ratio (rf=5%)"],
		cmap="viridis",
		s=10,
		alpha=0.65,
	)
	plt.colorbar(scatter, label="Sharpe ratio (risk-free rate 5%)")
	frontier = simulation.sort_values("Annualized Volatility").copy()
	frontier["Efficient Return"] = frontier["Annualized Return"].cummax()
	plt.plot(
		frontier["Annualized Volatility"] * 100,
		frontier["Efficient Return"] * 100,
		color="black",
		linewidth=1.8,
		label="Empirical efficient frontier",
		zorder=2,
	)
	plt.scatter(
		fund_volatility * 100,
		fund_return * 100,
		color="red",
		edgecolors="black",
		s=150,
		marker="*",
		label="Mutual fund observed portfolio",
		zorder=3,
	)
	plt.scatter(
		best_simulation["Annualized Volatility"] * 100,
		best_simulation["Annualized Return"] * 100,
		color="deepskyblue",
		edgecolors="black",
		s=220,
		marker="*",
		label="Best simulated portfolio",
		zorder=4,
	)
	plt.annotate(
		f"Best Portfolio\nSharpe: {best_simulation['Sharpe Ratio (rf=5%)']:.3f}",
		(best_simulation["Annualized Volatility"] * 100, best_simulation["Annualized Return"] * 100),
		xytext=(10, 12),
		textcoords="offset points",
		fontsize=9,
		bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "deepskyblue", "alpha": 0.9},
	)
	plt.xlabel("Annualized volatility (%)")
	plt.ylabel("Annualized total return (%)")
	plt.title("Efficient Frontier Simulation - Mutual Fund Portfolio")
	plt.grid(True, linestyle="--", alpha=0.35)
	plt.legend()
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "mutual_fund_efficient_frontier.png", dpi=300)
	plt.close()

	allocation_fig, allocation_ax = plt.subplots(figsize=(12, 7))
	bar_width = 0.38
	x = np.arange(len(TICKERS))
	allocation_ax.bar(
		x - bar_width / 2,
		allocation_comparison["Observed Top-10 Weight (%)"],
		width=bar_width,
		label="Observed Top-10 weights",
		color="darkorange",
	)
	allocation_ax.bar(
		x + bar_width / 2,
		allocation_comparison["Maximum Sharpe Weight (%)"],
		width=bar_width,
		label="Maximum Sharpe weights",
		color="steelblue",
	)
	allocation_ax.axhline(
		MAX_WEIGHT * 100,
		color="crimson",
		linestyle="--",
		linewidth=2,
		label="Maximum allocation constraint (13.94%)",
	)
	allocation_ax.set_xticks(x)
	allocation_ax.set_xticklabels(
		[ticker.replace(".JK", "") for ticker in TICKERS],
		rotation=45,
		ha="right",
	)
	allocation_ax.set_ylabel("Weight (%)")
	allocation_ax.set_title("Observed Top-10 vs Maximum Sharpe Portfolio Weights")
	allocation_ax.set_ylim(0, max(MAX_WEIGHT * 100 + 1, allocation_comparison.iloc[:, 1:].to_numpy().max() + 1))
	allocation_ax.grid(axis="y", linestyle="--", alpha=0.35)
	allocation_ax.legend()
	allocation_fig.tight_layout()
	allocation_fig.savefig(OUTPUT_DIR / "observed_vs_maximum_sharpe_weights.png", dpi=300)
	plt.close(allocation_fig)

	correlation = daily_returns.corr().reindex(index=TICKERS, columns=TICKERS)
	correlation.to_csv(OUTPUT_DIR / "mutual_fund_return_correlation.csv")

	heatmap_fig, heatmap_ax = plt.subplots(figsize=(10, 8))
	heatmap = heatmap_ax.imshow(correlation.to_numpy(), cmap="RdYlBu_r", vmin=-1, vmax=1)
	heatmap_fig.colorbar(heatmap, ax=heatmap_ax, label="Correlation")
	heatmap_ax.set_xticks(np.arange(len(TICKERS)))
	heatmap_ax.set_yticks(np.arange(len(TICKERS)))
	heatmap_ax.set_xticklabels(
		[ticker.replace(".JK", "") for ticker in TICKERS], rotation=45, ha="right"
	)
	heatmap_ax.set_yticklabels([ticker.replace(".JK", "") for ticker in TICKERS])
	for row in range(len(TICKERS)):
		for column in range(len(TICKERS)):
			value = correlation.iloc[row, column]
			heatmap_ax.text(
				column,
				row,
				f"{value:.2f}",
				ha="center",
				va="center",
				color="black" if abs(value) < 0.65 else "white",
				fontsize=8,
			)
	heatmap_ax.set_title("Daily Return Correlation Heatmap - Mutual Fund Stocks")
	heatmap_fig.tight_layout()
	heatmap_fig.savefig(OUTPUT_DIR / "mutual_fund_correlation_heatmap.png", dpi=300)
	plt.close(heatmap_fig)

	observed_daily_return = daily_returns.to_numpy() @ fund_weights
	best_daily_return = daily_returns[TICKERS].to_numpy() @ best_simulation[TICKERS].to_numpy(dtype=float)
	portfolio_returns = pd.DataFrame(
		{
			"Observed Top-10 Portfolio": observed_daily_return,
			"Best Portfolio": best_daily_return,
		},
		index=daily_returns.index,
	)
	cumulative_returns = (1 + portfolio_returns).cumprod() * 100
	cumulative_returns.index.name = "Date"
	cumulative_returns.to_csv(OUTPUT_DIR / "portfolio_cumulative_returns.csv")

	return_fig, return_ax = plt.subplots(figsize=(12, 7))
	return_ax.plot(
		cumulative_returns.index,
		cumulative_returns["Observed Top-10 Portfolio"],
		color="darkorange",
		linewidth=2,
		label="Observed Top-10 portfolio",
	)
	return_ax.plot(
		cumulative_returns.index,
		cumulative_returns["Best Portfolio"],
		color="steelblue",
		linewidth=2,
		label="Best portfolio",
	)
	return_ax.axhline(100, color="gray", linestyle=":", linewidth=1)
	return_ax.set_xlabel("Date")
	return_ax.set_ylabel("Portfolio value (initial = 100)")
	return_ax.set_title("Cumulative Total Return: Observed Top-10 vs Best Portfolio")
	return_ax.grid(True, linestyle="--", alpha=0.35)
	return_ax.legend()
	return_fig.tight_layout()
	return_fig.savefig(OUTPUT_DIR / "observed_vs_best_cumulative_return.png", dpi=300)
	plt.close(return_fig)

	print("=== Efficient Frontier Mutual Fund ===")
	print(f"Periode data: {prices.index.min().date()} sampai {prices.index.max().date()}")
	print("Harga yang digunakan: Adj Close (dividen direinvestasikan sebagai total return)")
	print(f"Jumlah simulasi: {NUM_SIMULATIONS:,}")
	print(f"Batas bobot per saham: {MAX_WEIGHT:.2%}")
	print(f"Risk-free rate: {RISK_FREE_RATE:.2%}")
	print("\nPerbandingan portofolio:")
	display = comparison.copy()
	display["Annualized Return"] *= 100
	display["Annualized Volatility"] *= 100
	print(display.round(4).to_string(index=False))
	print("\nKomposisi best simulated portfolio:")
	print(best_composition[["Ticker", "Best Portfolio Weight (%)"]].round(4).to_string(index=False))
	print(f"\nGrafik tersimpan: {OUTPUT_DIR / 'mutual_fund_efficient_frontier.png'}")
	print(f"Grafik bobot tersimpan: {OUTPUT_DIR / 'observed_vs_maximum_sharpe_weights.png'}")
	print(f"Heatmap korelasi tersimpan: {OUTPUT_DIR / 'mutual_fund_correlation_heatmap.png'}")
	print(f"Grafik cumulative return tersimpan: {OUTPUT_DIR / 'observed_vs_best_cumulative_return.png'}")
	print(f"Data simulasi tersimpan: {OUTPUT_DIR / 'mutual_fund_10000_simulations.csv'}")
	print(f"Komposisi best portfolio tersimpan: {OUTPUT_DIR / 'best_portfolio_composition.csv'}")


if __name__ == "__main__":
	main()