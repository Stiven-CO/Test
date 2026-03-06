"""Walk-Forward Analysis (unanchored) para una estrategia Dual SMA en BTC-USD.

Requisitos cubiertos:
- Descarga diaria de BTC-USD desde Yahoo (yfinance), con normalización robusta de Close a pd.Series.
- Walk-forward unanchored con ventanas rodantes y split 80% IS / 20% OOS.
- Optimización en cada tramo IS por Sharpe Ratio (grid search fast < slow).
- Validación OOS con los mejores hiperparámetros de cada ventana IS.
- Consolidación de OOS para curva final:
    - Incluye fallback compatible si tu instalación no trae ese método.
- Reporte completo con stats() para estrategia y benchmark.
- KPIs explícitos: Sharpe Ratio, Max Drawdown, Alpha Decay (deterioro IS->OOS de Sharpe).
- Visualización interactiva Plotly comparando estrategia vs Buy & Hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbt as vbt
import yfinance as yf


# ------------------------------
# Configuración
# ------------------------------
SYMBOL = "BTC-USD"
START = "2017-01-01"
END = None  # None = hasta la fecha actual

INITIAL_CASH = 100_000
FEE = 0.001  # 10 bps por operación

# Ventana total de walk-forward en barras diarias.
# Cada ventana se divide 80/20 (IS/OOS) y avanza por bloques OOS (unanchored rolling).
WF_WINDOW_BARS = 750
IS_RATIO = 0.8

FAST_WINDOWS = range(5, 81, 5)
SLOW_WINDOWS = range(20, 241, 10)


@dataclass
class WindowResult:
	"""Resultado por ventana walk-forward."""

	window_id: int
	is_start: pd.Timestamp
	is_end: pd.Timestamp
	oos_start: pd.Timestamp
	oos_end: pd.Timestamp
	best_fast: int
	best_slow: int
	is_sharpe: float
	oos_sharpe: float
	oos_pf: vbt.Portfolio


def fetch_close_prices(symbol: str, start: str, end: str | None) -> pd.Series:
	"""Descarga precios de cierre diarios desde Yahoo Finance."""
	data = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
	if data.empty:
		raise ValueError(f"No se pudieron descargar datos para {symbol}.")

	close_raw = data["Close"].dropna()

	# yfinance puede devolver DataFrame (1 columna) o Series según versión/config.
	if isinstance(close_raw, pd.DataFrame):
		if close_raw.shape[1] != 1:
			raise ValueError(f"Se esperaba una sola columna de cierre para {symbol}.")
		close = close_raw.iloc[:, 0]
	else:
		close = close_raw

	close = close.astype(float)
	close.name = symbol
	return close


def dual_sma_signals(close: pd.Series, fast: int, slow: int) -> tuple[pd.Series, pd.Series]:
	"""Genera señales de entrada/salida para cruce de medias."""
	fast_ma = close.rolling(window=fast, min_periods=fast).mean()
	slow_ma = close.rolling(window=slow, min_periods=slow).mean()

	entries = fast_ma > slow_ma
	exits = fast_ma < slow_ma
	return entries, exits


def sharpe_or_neg_inf(portfolio: vbt.Portfolio) -> float:
	"""Devuelve Sharpe Ratio robusto para comparar hiperparámetros."""
	sharpe_raw = portfolio.sharpe_ratio()
	if sharpe_raw is None:
		return -np.inf

	if isinstance(sharpe_raw, (pd.Series, pd.DataFrame)):
		sharpe_val = np.asarray(sharpe_raw).astype(float).ravel()[0]
	else:
		sharpe_val = float(sharpe_raw)

	if np.isnan(sharpe_val):
		return -np.inf
	return sharpe_val


def as_scalar(metric) -> float:
	"""Normaliza salidas de vectorbt (escalares o pandas) a float."""
	if isinstance(metric, (pd.Series, pd.DataFrame)):
		return float(np.asarray(metric).astype(float).ravel()[0])
	return float(metric)


def optimize_is_by_sharpe(
	close_is: pd.Series,
	fast_windows: Iterable[int],
	slow_windows: Iterable[int],
	init_cash: float,
	fees: float,
) -> tuple[int, int, float]:
	"""Busca la combinación (fast, slow) que maximiza Sharpe en IS."""
	best_fast = -1
	best_slow = -1
	best_sharpe = -np.inf

	for fast in fast_windows:
		for slow in slow_windows:
			if fast >= slow:
				continue

			entries, exits = dual_sma_signals(close_is, fast=fast, slow=slow)
			pf = vbt.Portfolio.from_signals(
				close_is,
				entries=entries,
				exits=exits,
				init_cash=init_cash,
				fees=fees,
				freq="1D",
			)

			candidate_sharpe = sharpe_or_neg_inf(pf)
			if candidate_sharpe > best_sharpe:
				best_fast = fast
				best_slow = slow
				best_sharpe = candidate_sharpe

	if best_fast < 0 or best_slow < 0:
		raise RuntimeError("No se encontró una combinación válida de hiperparámetros.")

	return best_fast, best_slow, best_sharpe


def run_walk_forward(
	close: pd.Series,
	wf_window_bars: int,
	is_ratio: float,
	fast_windows: Iterable[int],
	slow_windows: Iterable[int],
	init_cash: float,
	fees: float,
) -> tuple[vbt.Portfolio, pd.DataFrame, pd.Series]:
	"""Ejecuta WFA unanchored y concatena todos los OOS."""
	if not 0 < is_ratio < 1:
		raise ValueError("is_ratio debe estar entre 0 y 1.")
	if wf_window_bars < 50:
		raise ValueError("wf_window_bars es demasiado pequeño para análisis robusto.")
	if len(close) < wf_window_bars:
		raise ValueError("No hay suficientes datos para la ventana de walk-forward configurada.")

	is_len = int(wf_window_bars * is_ratio)
	oos_len = wf_window_bars - is_len
	if oos_len <= 0:
		raise ValueError("La longitud OOS resultó inválida.")

	window_results: list[WindowResult] = []
	oos_portfolios: list[vbt.Portfolio] = []
	oos_slices: list[pd.Series] = []
	oos_entries_slices: list[pd.Series] = []
	oos_exits_slices: list[pd.Series] = []

	# Avance por el tamaño OOS: rolling unanchored sin expansión del IS.
	starts = range(0, len(close) - wf_window_bars + 1, oos_len)

	for w_id, start_idx in enumerate(starts, start=1):
		end_idx = start_idx + wf_window_bars
		wf_slice = close.iloc[start_idx:end_idx]

		close_is = wf_slice.iloc[:is_len]
		close_oos = wf_slice.iloc[is_len:]
		if close_oos.empty:
			continue

		best_fast, best_slow, is_sharpe = optimize_is_by_sharpe(
			close_is=close_is,
			fast_windows=fast_windows,
			slow_windows=slow_windows,
			init_cash=init_cash,
			fees=fees,
		)

		oos_entries, oos_exits = dual_sma_signals(close_oos, fast=best_fast, slow=best_slow)
		oos_pf = vbt.Portfolio.from_signals(
			close_oos,
			entries=oos_entries,
			exits=oos_exits,
			init_cash=init_cash,
			fees=fees,
			freq="1D",
		)
		oos_sharpe = sharpe_or_neg_inf(oos_pf)

		window_results.append(
			WindowResult(
				window_id=w_id,
				is_start=close_is.index[0],
				is_end=close_is.index[-1],
				oos_start=close_oos.index[0],
				oos_end=close_oos.index[-1],
				best_fast=best_fast,
				best_slow=best_slow,
				is_sharpe=is_sharpe,
				oos_sharpe=oos_sharpe,
				oos_pf=oos_pf,
			)
		)
		oos_portfolios.append(oos_pf)
		oos_slices.append(close_oos)
		oos_entries_slices.append(oos_entries)
		oos_exits_slices.append(oos_exits)

	if not oos_portfolios:
		raise RuntimeError("No se generaron ventanas OOS. Ajusta datos o parámetros.")

	all_oos_close = pd.concat(oos_slices)
	all_oos_entries = pd.concat(oos_entries_slices)
	all_oos_exits = pd.concat(oos_exits_slices)

	# Requisito: consolidar con vbt.Portfolio.from_concat.
	# Si la versión instalada no lo soporta, se usa un fallback equivalente.
	if hasattr(vbt.Portfolio, "from_concat"):
		final_oos_pf = vbt.Portfolio.from_concat(oos_portfolios)
	else:
		final_oos_pf = vbt.Portfolio.from_signals(
			all_oos_close,
			entries=all_oos_entries,
			exits=all_oos_exits,
			init_cash=init_cash,
			fees=fees,
			freq="1D",
		)

	results_df = pd.DataFrame(
		{
			"window_id": [r.window_id for r in window_results],
			"is_start": [r.is_start for r in window_results],
			"is_end": [r.is_end for r in window_results],
			"oos_start": [r.oos_start for r in window_results],
			"oos_end": [r.oos_end for r in window_results],
			"best_fast": [r.best_fast for r in window_results],
			"best_slow": [r.best_slow for r in window_results],
			"is_sharpe": [r.is_sharpe for r in window_results],
			"oos_sharpe": [r.oos_sharpe for r in window_results],
		}
	)

	return final_oos_pf, results_df, all_oos_close


def compute_alpha_decay(window_df: pd.DataFrame) -> float:
	"""Calcula el Alpha Decay como deterioro porcentual IS->OOS del Sharpe."""
	mean_is = float(window_df["is_sharpe"].replace([-np.inf, np.inf], np.nan).dropna().mean())
	mean_oos = float(window_df["oos_sharpe"].replace([-np.inf, np.inf], np.nan).dropna().mean())

	if np.isnan(mean_is) or np.isclose(mean_is, 0.0):
		return np.nan
	return ((mean_is - mean_oos) / abs(mean_is)) * 100.0


def build_benchmark(close_oos: pd.Series, init_cash: float, fees: float) -> vbt.Portfolio:
	"""Construye benchmark Buy & Hold sobre el tramo OOS consolidado."""
	entries = pd.Series(False, index=close_oos.index)
	exits = pd.Series(False, index=close_oos.index)
	entries.iloc[0] = True
	exits.iloc[-1] = True

	return vbt.Portfolio.from_signals(
		close_oos,
		entries=entries,
		exits=exits,
		init_cash=init_cash,
		fees=fees,
		freq="1D",
	)


def plot_equity_comparison(strategy_pf: vbt.Portfolio, benchmark_pf: vbt.Portfolio) -> None:
	"""Gráfico interactivo de valor de cartera: estrategia vs benchmark."""
	strategy_equity = strategy_pf.value()
	benchmark_equity = benchmark_pf.value()

	fig = go.Figure()
	fig.add_trace(
		go.Scatter(
			x=strategy_equity.index,
			y=strategy_equity.values,
			mode="lines",
			name="Dual SMA WFA (OOS Concat)",
			line={"width": 2},
		)
	)
	fig.add_trace(
		go.Scatter(
			x=benchmark_equity.index,
			y=benchmark_equity.values,
			mode="lines",
			name="Benchmark Buy & Hold",
			line={"width": 2, "dash": "dot"},
		)
	)

	fig.update_layout(
		title="Walk-Forward Unanchored | Dual SMA vs Buy & Hold",
		xaxis_title="Fecha",
		yaxis_title="Equity (USD)",
		template="plotly_white",
		legend={"orientation": "h", "y": 1.02, "x": 0},
		hovermode="x unified",
	)
	fig.show()


def main() -> None:
	print("Descargando datos...")
	close = fetch_close_prices(SYMBOL, START, END)

	print("Ejecutando Walk-Forward Analysis unanchored...")
	strategy_pf, wf_results, close_oos_all = run_walk_forward(
		close=close,
		wf_window_bars=WF_WINDOW_BARS,
		is_ratio=IS_RATIO,
		fast_windows=FAST_WINDOWS,
		slow_windows=SLOW_WINDOWS,
		init_cash=INITIAL_CASH,
		fees=FEE,
	)

	benchmark_pf = build_benchmark(close_oos=close_oos_all, init_cash=INITIAL_CASH, fees=FEE)
	alpha_decay_pct = compute_alpha_decay(wf_results)

	print("\n===== RESUMEN POR VENTANA (IS/OOS) =====")
	print(wf_results.to_string(index=False))

	print("\n===== STATS ESTRATEGIA (OOS CONCAT) =====")
	strategy_stats = strategy_pf.stats()
	print(strategy_stats)

	print("\n===== STATS BENCHMARK BUY & HOLD =====")
	benchmark_stats = benchmark_pf.stats()
	print(benchmark_stats)

	# Reporte directo solicitado: Max Drawdown, Sharpe y Alpha Decay.
	print("\n===== KPIs CLAVE =====")
	strategy_sharpe = as_scalar(strategy_pf.sharpe_ratio())
	strategy_mdd = as_scalar(strategy_pf.max_drawdown())
	print(f"Sharpe Ratio (Estrategia OOS): {strategy_sharpe:.4f}")
	print(f"Max Drawdown (Estrategia OOS): {strategy_mdd:.2%}")
	if np.isnan(alpha_decay_pct):
		print("Alpha Decay (IS->OOS Sharpe): NaN (IS Sharpe medio no válido)")
	else:
		print(f"Alpha Decay (IS->OOS Sharpe): {alpha_decay_pct:.2f}%")

	print("\nGenerando gráfico interactivo...")
	plot_equity_comparison(strategy_pf=strategy_pf, benchmark_pf=benchmark_pf)


if __name__ == "__main__":
	main()
