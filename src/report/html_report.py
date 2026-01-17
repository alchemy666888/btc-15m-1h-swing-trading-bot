"""
HTML Report Generator with Interactive Charts using Plotly
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from jinja2 import Template
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.backtest.monte_carlo import MonteCarloResult


class HTMLReportGenerator:
    """Generate interactive HTML reports for backtest results"""

    def __init__(self, output_dir: str = config.REPORTS_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self,
        backtest_results: Dict,
        monte_carlo_result: Optional[MonteCarloResult],
        price_data: pd.DataFrame,
        benchmark: Dict,
        report_name: str = "backtest_report"
    ) -> str:
        """
        Generate comprehensive HTML report.

        Args:
            backtest_results: Results from BacktestEngine
            monte_carlo_result: Results from Monte Carlo simulation
            price_data: Price DataFrame for charts
            benchmark: Buy & hold benchmark results
            report_name: Name for the report file

        Returns:
            Path to generated HTML file
        """
        # Generate all chart HTML
        equity_chart = self._create_equity_chart(backtest_results, benchmark)
        drawdown_chart = self._create_drawdown_chart(backtest_results)
        price_chart = self._create_price_chart(price_data, backtest_results.get('trade_log', []))
        trade_distribution = self._create_trade_distribution(backtest_results)
        monthly_returns = self._create_monthly_returns_heatmap(backtest_results)

        mc_charts = ""
        if monte_carlo_result:
            mc_charts = self._create_monte_carlo_charts(monte_carlo_result)

        # Generate HTML
        html_content = self._render_template(
            backtest_results=backtest_results,
            monte_carlo_result=monte_carlo_result,
            benchmark=benchmark,
            equity_chart=equity_chart,
            drawdown_chart=drawdown_chart,
            price_chart=price_chart,
            trade_distribution=trade_distribution,
            monthly_returns=monthly_returns,
            mc_charts=mc_charts
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_name}_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Report generated: {filepath}")
        return filepath

    def _create_equity_chart(self, results: Dict, benchmark: Dict) -> str:
        """Create equity curve chart"""
        equity_curve = results.get('equity_curve', [])
        if not equity_curve:
            return "<p>No equity curve data available</p>"

        dates = [e['date'] for e in equity_curve]
        values = [e['value'] for e in equity_curve]

        # Calculate benchmark equity curve
        initial = results['initial_capital']
        benchmark_return = benchmark['return_pct'] / 100
        benchmark_values = np.linspace(initial, initial * (1 + benchmark_return), len(values))

        fig = go.Figure()

        # Strategy equity
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Strategy',
            line=dict(color='#2ecc71', width=2)
        ))

        # Benchmark
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#3498db', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_dark',
            height=400
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_drawdown_chart(self, results: Dict) -> str:
        """Create drawdown chart"""
        equity_curve = results.get('equity_curve', [])
        if not equity_curve:
            return "<p>No drawdown data available</p>"

        dates = [e['date'] for e in equity_curve]
        values = np.array([e['value'] for e in equity_curve])

        # Calculate drawdown
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#e74c3c', width=1),
            fillcolor='rgba(231, 76, 60, 0.3)'
        ))

        fig.update_layout(
            title='Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_dark',
            height=300
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_price_chart(self, price_data: pd.DataFrame, trade_log: List) -> str:
        """Create candlestick chart with trade markers"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_data['timestamp'],
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='BTCUSDT'
            ),
            row=1, col=1
        )

        # Add trade markers
        if trade_log:
            long_entries = [t for t in trade_log if t.get('direction') == 'long']
            short_entries = [t for t in trade_log if t.get('direction') == 'short']

            if long_entries:
                fig.add_trace(
                    go.Scatter(
                        x=[t.get('entry_datetime', t.get('entry_date', '')) for t in long_entries],
                        y=[t['entry_price'] for t in long_entries],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=12, color='#2ecc71'),
                        name='Long Entry'
                    ),
                    row=1, col=1
                )

            if short_entries:
                fig.add_trace(
                    go.Scatter(
                        x=[t.get('entry_datetime', t.get('entry_date', '')) for t in short_entries],
                        y=[t['entry_price'] for t in short_entries],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=12, color='#e74c3c'),
                        name='Short Entry'
                    ),
                    row=1, col=1
                )

        # Volume chart
        colors = ['#2ecc71' if close >= open_ else '#e74c3c'
                  for close, open_ in zip(price_data['close'], price_data['open'])]

        fig.add_trace(
            go.Bar(
                x=price_data['timestamp'],
                y=price_data['volume'],
                marker_color=colors,
                name='Volume',
                showlegend=False
            ),
            row=2, col=1
        )

        fig.update_layout(
            title='Price Chart with Trade Entries',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600,
            showlegend=True
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_trade_distribution(self, results: Dict) -> str:
        """Create trade P&L distribution histogram"""
        trade_log = results.get('trade_log', [])
        if not trade_log:
            return "<p>No trade data available</p>"

        pnls = [t['pnl'] for t in trade_log]

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=pnls,
            nbinsx=30,
            marker_color=['#2ecc71' if p > 0 else '#e74c3c' for p in sorted(pnls)],
            name='Trade P&L'
        ))

        fig.update_layout(
            title='Trade P&L Distribution',
            xaxis_title='P&L ($)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=350
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_monthly_returns_heatmap(self, results: Dict) -> str:
        """Create monthly returns heatmap"""
        time_returns = results.get('time_returns', {})
        if not time_returns:
            return "<p>No monthly return data available</p>"

        # Convert to DataFrame
        df = pd.DataFrame([
            {'date': date, 'return': ret * 100}
            for date, ret in time_returns.items()
        ])

        if df.empty:
            return "<p>No monthly return data available</p>"

        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

        # Aggregate by month
        monthly = df.groupby(['year', 'month'])['return'].sum().reset_index()
        monthly_pivot = monthly.pivot(index='year', columns='month', values='return')

        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig = go.Figure(data=go.Heatmap(
            z=monthly_pivot.values,
            x=month_names[:monthly_pivot.shape[1]],
            y=monthly_pivot.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(monthly_pivot.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            title='Monthly Returns (%)',
            template='plotly_dark',
            height=300
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_monte_carlo_charts(self, mc_result: MonteCarloResult) -> str:
        """Create Monte Carlo simulation charts"""
        charts_html = ""

        # 1. Equity curves fan chart
        fig1 = go.Figure()

        # Plot sample of equity curves
        n_curves_to_plot = min(100, len(mc_result.equity_curves))
        indices = np.random.choice(len(mc_result.equity_curves), n_curves_to_plot, replace=False)

        for idx in indices:
            curve = mc_result.equity_curves[idx]
            fig1.add_trace(go.Scatter(
                y=curve,
                mode='lines',
                line=dict(color='rgba(52, 152, 219, 0.2)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add percentile curves
        all_curves = np.array([c for c in mc_result.equity_curves if len(c) == len(mc_result.equity_curves[0])])
        if len(all_curves) > 0:
            percentile_5 = np.percentile(all_curves, 5, axis=0)
            percentile_50 = np.percentile(all_curves, 50, axis=0)
            percentile_95 = np.percentile(all_curves, 95, axis=0)

            fig1.add_trace(go.Scatter(
                y=percentile_95,
                mode='lines',
                name='95th Percentile',
                line=dict(color='#2ecc71', width=2)
            ))
            fig1.add_trace(go.Scatter(
                y=percentile_50,
                mode='lines',
                name='Median',
                line=dict(color='#f39c12', width=2)
            ))
            fig1.add_trace(go.Scatter(
                y=percentile_5,
                mode='lines',
                name='5th Percentile',
                line=dict(color='#e74c3c', width=2)
            ))

        fig1.update_layout(
            title='Monte Carlo Equity Curves',
            xaxis_title='Trade #',
            yaxis_title='Portfolio Value ($)',
            template='plotly_dark',
            height=400
        )

        charts_html += '<div class="chart-container">'
        charts_html += fig1.to_html(full_html=False, include_plotlyjs=False)
        charts_html += '</div>'

        # 2. Final returns distribution
        fig2 = go.Figure()

        returns = (mc_result.final_values - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL * 100

        fig2.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            marker_color='#3498db',
            name='Returns'
        ))

        # Add vertical lines for key percentiles
        fig2.add_vline(x=mc_result.percentile_5, line_dash="dash",
                       line_color="#e74c3c", annotation_text="5%")
        fig2.add_vline(x=mc_result.median_return, line_dash="dash",
                       line_color="#f39c12", annotation_text="Median")
        fig2.add_vline(x=mc_result.percentile_95, line_dash="dash",
                       line_color="#2ecc71", annotation_text="95%")

        fig2.update_layout(
            title='Monte Carlo Return Distribution',
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=350
        )

        charts_html += '<div class="chart-container">'
        charts_html += fig2.to_html(full_html=False, include_plotlyjs=False)
        charts_html += '</div>'

        # 3. Max Drawdown distribution
        fig3 = go.Figure()

        fig3.add_trace(go.Histogram(
            x=mc_result.max_drawdowns,
            nbinsx=50,
            marker_color='#e74c3c',
            name='Max Drawdown'
        ))

        fig3.update_layout(
            title='Monte Carlo Max Drawdown Distribution',
            xaxis_title='Max Drawdown (%)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=350
        )

        charts_html += '<div class="chart-container">'
        charts_html += fig3.to_html(full_html=False, include_plotlyjs=False)
        charts_html += '</div>'

        return charts_html

    def _render_template(
        self,
        backtest_results: Dict,
        monte_carlo_result: Optional[MonteCarloResult],
        benchmark: Dict,
        equity_chart: str,
        drawdown_chart: str,
        price_chart: str,
        trade_distribution: str,
        monthly_returns: str,
        mc_charts: str
    ) -> str:
        """Render the HTML template"""

        template = Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - BTC Swing Trading Strategy</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-tertiary: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent-green: #2ecc71;
            --accent-red: #e74c3c;
            --accent-blue: #3498db;
            --accent-yellow: #f39c12;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }

        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        header p {
            color: var(--text-secondary);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid var(--accent-blue);
        }

        .metric-card.positive {
            border-left-color: var(--accent-green);
        }

        .metric-card.negative {
            border-left-color: var(--accent-red);
        }

        .metric-card h3 {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-card .value {
            font-size: 1.8rem;
            font-weight: bold;
        }

        .metric-card .sub-value {
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .section {
            background: var(--bg-secondary);
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
        }

        .section h2 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--bg-tertiary);
        }

        .chart-container {
            margin: 20px 0;
        }

        .two-col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 768px) {
            .two-col {
                grid-template-columns: 1fr;
            }
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--bg-tertiary);
        }

        th {
            background: var(--bg-tertiary);
            font-weight: 600;
        }

        tr:hover {
            background: rgba(52, 152, 219, 0.1);
        }

        .positive-text { color: var(--accent-green); }
        .negative-text { color: var(--accent-red); }
        .neutral-text { color: var(--accent-yellow); }

        .summary-box {
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>BTC Swing Trading Strategy</h1>
            <p>Backtest Report - {{ backtest_results.get('start_date', 'N/A') }} to {{ backtest_results.get('end_date', 'N/A') }}</p>
            <p>Generated: {{ generation_time }}</p>
        </header>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card {% if backtest_results['total_return_pct'] > 0 %}positive{% else %}negative{% endif %}">
                <h3>Total Return</h3>
                <div class="value {% if backtest_results['total_return_pct'] > 0 %}positive-text{% else %}negative-text{% endif %}">
                    {{ "%.2f"|format(backtest_results['total_return_pct']) }}%
                </div>
                <div class="sub-value">${{ "{:,.2f}".format(backtest_results['total_return_usd']) }}</div>
            </div>

            <div class="metric-card">
                <h3>Final Value</h3>
                <div class="value">${{ "{:,.2f}".format(backtest_results['final_value']) }}</div>
                <div class="sub-value">Initial: ${{ "{:,.2f}".format(backtest_results['initial_capital']) }}</div>
            </div>

            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <div class="value {% if backtest_results['sharpe_ratio'] > 1 %}positive-text{% elif backtest_results['sharpe_ratio'] < 0 %}negative-text{% else %}neutral-text{% endif %}">
                    {{ "%.2f"|format(backtest_results['sharpe_ratio']) }}
                </div>
            </div>

            <div class="metric-card negative">
                <h3>Max Drawdown</h3>
                <div class="value negative-text">{{ "%.2f"|format(backtest_results['max_drawdown_pct']) }}%</div>
                <div class="sub-value">${{ "{:,.2f}".format(backtest_results['max_drawdown_usd']) }}</div>
            </div>

            <div class="metric-card">
                <h3>Win Rate</h3>
                <div class="value {% if backtest_results['win_rate'] > 50 %}positive-text{% else %}negative-text{% endif %}">
                    {{ "%.1f"|format(backtest_results['win_rate']) }}%
                </div>
                <div class="sub-value">{{ backtest_results['won_trades'] }}/{{ backtest_results['total_trades'] }} trades</div>
            </div>

            <div class="metric-card">
                <h3>Profit Factor</h3>
                <div class="value {% if backtest_results['profit_factor'] > 1 %}positive-text{% else %}negative-text{% endif %}">
                    {{ "%.2f"|format(backtest_results['profit_factor']) }}
                </div>
            </div>
        </div>

        <!-- Equity & Drawdown Charts -->
        <div class="section">
            <h2>Performance Overview</h2>
            <div class="chart-container">
                {{ equity_chart|safe }}
            </div>
            <div class="chart-container">
                {{ drawdown_chart|safe }}
            </div>
        </div>

        <!-- Price Chart -->
        <div class="section">
            <h2>Price Chart with Trades</h2>
            <div class="chart-container">
                {{ price_chart|safe }}
            </div>
        </div>

        <!-- Trade Analysis -->
        <div class="section">
            <h2>Trade Analysis</h2>
            <div class="two-col">
                <div class="chart-container">
                    {{ trade_distribution|safe }}
                </div>
                <div class="chart-container">
                    {{ monthly_returns|safe }}
                </div>
            </div>

            <div class="summary-box">
                <h3>Trade Statistics</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Trades</td>
                        <td>{{ backtest_results['total_trades'] }}</td>
                    </tr>
                    <tr>
                        <td>Won Trades</td>
                        <td class="positive-text">{{ backtest_results['won_trades'] }}</td>
                    </tr>
                    <tr>
                        <td>Lost Trades</td>
                        <td class="negative-text">{{ backtest_results['lost_trades'] }}</td>
                    </tr>
                    <tr>
                        <td>Average Win</td>
                        <td class="positive-text">${{ "{:,.2f}".format(backtest_results['avg_win']) }}</td>
                    </tr>
                    <tr>
                        <td>Average Loss</td>
                        <td class="negative-text">${{ "{:,.2f}".format(backtest_results['avg_loss']) }}</td>
                    </tr>
                    <tr>
                        <td>Average Trade</td>
                        <td>${{ "{:,.2f}".format(backtest_results['avg_trade']) }}</td>
                    </tr>
                    <tr>
                        <td>Longest Win Streak</td>
                        <td>{{ backtest_results['longest_winning_streak'] }}</td>
                    </tr>
                    <tr>
                        <td>Longest Loss Streak</td>
                        <td>{{ backtest_results['longest_losing_streak'] }}</td>
                    </tr>
                </table>
            </div>
        </div>

        <!-- Benchmark Comparison -->
        <div class="section">
            <h2>Benchmark Comparison</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Strategy</th>
                    <th>Buy & Hold</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td class="{% if backtest_results['total_return_pct'] > 0 %}positive-text{% else %}negative-text{% endif %}">
                        {{ "%.2f"|format(backtest_results['total_return_pct']) }}%
                    </td>
                    <td class="{% if benchmark['return_pct'] > 0 %}positive-text{% else %}negative-text{% endif %}">
                        {{ "%.2f"|format(benchmark['return_pct']) }}%
                    </td>
                    <td class="{% if backtest_results['total_return_pct'] > benchmark['return_pct'] %}positive-text{% else %}negative-text{% endif %}">
                        {{ "%.2f"|format(backtest_results['total_return_pct'] - benchmark['return_pct']) }}%
                    </td>
                </tr>
                <tr>
                    <td>Final Value</td>
                    <td>${{ "{:,.2f}".format(backtest_results['final_value']) }}</td>
                    <td>${{ "{:,.2f}".format(benchmark['final_value']) }}</td>
                    <td>${{ "{:,.2f}".format(backtest_results['final_value'] - benchmark['final_value']) }}</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td class="negative-text">{{ "%.2f"|format(backtest_results['max_drawdown_pct']) }}%</td>
                    <td class="negative-text">{{ "%.2f"|format(benchmark['max_drawdown']) }}%</td>
                    <td>{{ "%.2f"|format(backtest_results['max_drawdown_pct'] - benchmark['max_drawdown']) }}%</td>
                </tr>
            </table>
        </div>

        {% if monte_carlo_result %}
        <!-- Monte Carlo Analysis -->
        <div class="section">
            <h2>Monte Carlo Analysis</h2>
            <p style="color: var(--text-secondary); margin-bottom: 20px;">
                {{ n_simulations }} simulations with trade resampling to assess strategy robustness.
            </p>

            {{ mc_charts|safe }}

            <div class="summary-box">
                <h3>Simulation Statistics</h3>
                <div class="two-col">
                    <table>
                        <tr>
                            <th>Return Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Mean Return</td>
                            <td>{{ "%.2f"|format(monte_carlo_result.mean_return) }}%</td>
                        </tr>
                        <tr>
                            <td>Median Return</td>
                            <td>{{ "%.2f"|format(monte_carlo_result.median_return) }}%</td>
                        </tr>
                        <tr>
                            <td>Std Deviation</td>
                            <td>{{ "%.2f"|format(monte_carlo_result.std_return) }}%</td>
                        </tr>
                        <tr>
                            <td>5th Percentile</td>
                            <td class="negative-text">{{ "%.2f"|format(monte_carlo_result.percentile_5) }}%</td>
                        </tr>
                        <tr>
                            <td>95th Percentile</td>
                            <td class="positive-text">{{ "%.2f"|format(monte_carlo_result.percentile_95) }}%</td>
                        </tr>
                    </table>
                    <table>
                        <tr>
                            <th>Risk Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Probability of Profit</td>
                            <td class="{% if monte_carlo_result.probability_profit > 50 %}positive-text{% else %}negative-text{% endif %}">
                                {{ "%.1f"|format(monte_carlo_result.probability_profit) }}%
                            </td>
                        </tr>
                        <tr>
                            <td>VaR (95%)</td>
                            <td class="negative-text">{{ "%.2f"|format(monte_carlo_result.var_95) }}%</td>
                        </tr>
                        <tr>
                            <td>CVaR (95%)</td>
                            <td class="negative-text">{{ "%.2f"|format(monte_carlo_result.cvar_95) }}%</td>
                        </tr>
                        <tr>
                            <td>Mean Max Drawdown</td>
                            <td class="negative-text">{{ "%.2f"|format(monte_carlo_result.mean_max_drawdown) }}%</td>
                        </tr>
                        <tr>
                            <td>Worst Max Drawdown</td>
                            <td class="negative-text">{{ "%.2f"|format(monte_carlo_result.worst_max_drawdown) }}%</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Risk Metrics -->
        <div class="section">
            <h2>Risk Metrics Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{{ "%.2f"|format(backtest_results['sharpe_ratio']) }}</td>
                    <td>{% if backtest_results['sharpe_ratio'] > 2 %}Excellent{% elif backtest_results['sharpe_ratio'] > 1 %}Good{% elif backtest_results['sharpe_ratio'] > 0 %}Acceptable{% else %}Poor{% endif %}</td>
                </tr>
                <tr>
                    <td>Sortino Ratio</td>
                    <td>{{ "%.2f"|format(backtest_results['sortino_ratio']) }}</td>
                    <td>{% if backtest_results['sortino_ratio'] > 2 %}Excellent{% elif backtest_results['sortino_ratio'] > 1 %}Good{% elif backtest_results['sortino_ratio'] > 0 %}Acceptable{% else %}Poor{% endif %}</td>
                </tr>
                <tr>
                    <td>SQN (System Quality Number)</td>
                    <td>{{ "%.2f"|format(backtest_results['sqn']) }}</td>
                    <td>{% if backtest_results['sqn'] > 3 %}Excellent{% elif backtest_results['sqn'] > 2 %}Good{% elif backtest_results['sqn'] > 1.5 %}Above Average{% else %}Average{% endif %}</td>
                </tr>
                <tr>
                    <td>CAGR</td>
                    <td>{{ "%.2f"|format(backtest_results['cagr']) }}%</td>
                    <td>Compound Annual Growth Rate</td>
                </tr>
            </table>
        </div>

        <footer>
            <p>Report generated by BTC Swing Trading Backtester</p>
            <p>Strategy: Multi-timeframe Swing Trading | Timeframes: 1H + 15min | Leverage: {{ leverage }}x</p>
        </footer>
    </div>
</body>
</html>
        ''')

        return template.render(
            backtest_results=backtest_results,
            monte_carlo_result=monte_carlo_result,
            benchmark=benchmark,
            equity_chart=equity_chart,
            drawdown_chart=drawdown_chart,
            price_chart=price_chart,
            trade_distribution=trade_distribution,
            monthly_returns=monthly_returns,
            mc_charts=mc_charts,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_simulations=config.MONTE_CARLO_RUNS,
            leverage=config.LEVERAGE
        )


def main():
    """Test report generation"""
    print("HTML Report Generator module loaded successfully.")


if __name__ == "__main__":
    main()
