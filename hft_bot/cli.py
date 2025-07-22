"""Modern CLI with Typer"""

import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from hft_bot import __version__
from hft_bot.bot import HFTBot

# Initialize Typer app
app = typer.Typer(
    name="hft",
    help="🚀 HFT Bot - High Frequency Trading System",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()

def version_callback(value: bool):
    """Show version and exit"""
    if value:
        console.print(f"[bold blue]HFT Bot[/bold blue] version [green]{__version__}[/green]")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version and exit"
    ),
):
    """HFT Bot - Enterprise Ready Trading System"""
    pass

@app.command(name="run", help="🎯 Run the trading bot")
def run_command(
    config: Path = typer.Option(
        "config/bot_config.yaml",
        "--config", "-c",
        help="Configuration file path",
        exists=True,
    ),
    paper: bool = typer.Option(
        False,
        "--paper", "-p",
        help="Run in paper trading mode"
    ),
    testnet: bool = typer.Option(
        False,
        "--testnet", "-t",
        help="Use testnet for live trading"
    ),
    strategy: Optional[str] = typer.Option(
        None,
        "--strategy", "-s",
        help="Trading strategy to use"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level", "-l",
        help="Logging level",
        case_sensitive=False,
    ),
):
    """Run the trading bot"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Starting HFT Bot...", total=None)
        
        try:
            from hft_bot.commands.run import run_bot
            
            asyncio.run(run_bot(
                config_file=str(config),
                paper_trading=paper,
                testnet=testnet,
                strategy=strategy,
                log_level=log_level
            ))
        except KeyboardInterrupt:
            console.print("\n[yellow]Bot stopped by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            raise typer.Exit(1)

@app.command(name="backtest", help="📊 Run backtesting")
def backtest_command(
    data_file: Path = typer.Argument(
        ...,
        help="Historical data file (CSV)",
        exists=True,
    ),
    config: Path = typer.Option(
        "config/bot_config.yaml",
        "--config", "-c",
        help="Configuration file",
        exists=True,
    ),
    start_date: Optional[datetime] = typer.Option(
        None,
        "--start", "-s",
        help="Start date (YYYY-MM-DD)",
        formats=["%Y-%m-%d"],
    ),
    end_date: Optional[datetime] = typer.Option(
        None,
        "--end", "-e",
        help="End date (YYYY-MM-DD)",
        formats=["%Y-%m-%d"],
    ),
    output: Path = typer.Option(
        "backtest_results.json",
        "--output", "-o",
        help="Output file for results",
    ),
    show_plot: bool = typer.Option(
        True,
        "--plot/--no-plot",
        help="Show interactive plot of results"
    ),
    walk_forward: bool = typer.Option(
        False,
        "--walk-forward/--no-walk-forward",
        help="Use walk-forward analysis"
    ),
    window_test: int = typer.Option(
        30,
        "--test",
        help="Test window size in days for walk-forward"
    ),
):
    """Run backtesting on historical data"""
    console.print(f"[bold]Running backtest on {data_file.name}[/bold]")
    
    try:
        from hft_bot.commands.backtest import run_backtest
        
        results = asyncio.run(run_backtest(
            data_file=str(data_file),
            config_file=str(config),
            start_date=start_date,
            end_date=end_date,
            output_file=str(output),
            show_plot=show_plot,
            walk_forward=walk_forward,
            window_test=window_test
        ))
        
        # Display results table
        table = Table(title="Backtest Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        if walk_forward and 'periods' in results:
            table.add_row("Periods", str(results['periods']))
            table.add_row("Average Return", f"{results['avg_return']:.2%}")
            table.add_row("Average Sharpe", f"{results['avg_sharpe']:.2f}")
            table.add_row("Consistency", f"{results['consistency']:.1%}")
            table.add_row("Total Return", f"{results['total_return']:.2%}")
        else:
            table.add_row("Total Return", f"{results['total_return']:.2%}")
            table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            table.add_row("Max Drawdown", f"{results['max_drawdown']:.2%}")
            table.add_row("Win Rate", f"{results['win_rate']:.2%}")
            table.add_row("Total Trades", str(results['total_trades']))
        
        console.print(table)
        console.print(f"\n[green]Results saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Backtest failed: {e}[/red]")
        raise typer.Exit(1)

@app.command(name="optimize", help="🔧 Optimize trading parameters")
def optimize_command(
    data_file: Path = typer.Argument(
        ...,
        help="Historical data file (CSV)",
        exists=True,
    ),
    trials: int = typer.Option(
        100,
        "--trials", "-n",
        help="Number of optimization trials",
        min=10,
        max=10000,
    ),
    metric: str = typer.Option(
        "sharpe",
        "--metric", "-m",
        help="Metric to optimize",
        case_sensitive=False,
    ),
    jobs: int = typer.Option(
        -1,
        "--jobs", "-j",
        help="Number of parallel jobs (-1 for all CPUs)",
    ),
    study_name: Optional[str] = typer.Option(
        None,
        "--study", "-s",
        help="Optuna study name for resuming",
    ),
):
    """Optimize trading parameters using Optuna"""
    console.print(f"[bold]Optimizing {metric} with {trials} trials[/bold]")
    
    try:
        from hft_bot.commands.optimize import run_optimization
        
        best_params = asyncio.run(run_optimization(
            data_file=str(data_file),
            n_trials=trials,
            metric=metric,
            n_jobs=jobs,
            study_name=study_name
        ))
        
        # Display best parameters
        table = Table(title="Best Parameters", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        for param, value in best_params.items():
            table.add_row(param, str(value))
            
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        raise typer.Exit(1)

@app.command(name="init", help="🎉 Initialize new configuration")
def init_command(
    output: Path = typer.Option(
        "config/bot_config.yaml",
        "--output", "-o",
        help="Output configuration file",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive", "-i/-n",
        help="Interactive configuration mode",
    ),
):
    """Initialize a new configuration file"""
    if interactive:
        console.print("[bold]Welcome to HFT Bot Configuration![/bold]\n")
        
        # Interactive prompts
        symbol = typer.prompt("Trading symbol", default="BTCUSDT")
        paper = typer.confirm("Enable paper trading?", default=True)
        capital = typer.prompt("Initial capital", default=1000, type=float)
        
        config_content = f"""# HFT Bot Configuration
symbol: {symbol}
paper_trading: {str(paper).lower()}
initial_capital: {capital}

trading_params:
  decision_interval: 0.5
  min_signal_confluences: 2
  gross_take_profit: 0.0008
  gross_stop_loss: 0.0004
  max_position_size: 500
  kelly_fraction: 0.25

risk:
  max_drawdown: 0.15
  max_positions: 3
  max_exposure: 0.5
  daily_loss_limit: 0.05
  position_timeout: 600

alerts:
  email:
    enabled: false
"""
    else:
        # Use default template
        from hft_bot.templates import DEFAULT_CONFIG
        config_content = DEFAULT_CONFIG
    
    # Create directory if needed
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration
    output.write_text(config_content, encoding="utf-8")
    console.print(f"[green]Configuration created at {output}[/green]")

@app.command(name="status", help="📈 Show bot status")
def status_command(
    host: str = typer.Option(
        "localhost",
        "--host", "-h",
        help="Bot host address",
    ),
    port: int = typer.Option(
        8080,
        "--port", "-p",
        help="Bot metrics port",
    ),
):
    """Show current bot status and metrics"""
    import httpx
    
    try:
        response = httpx.get(f"http://{host}:{port}/status", timeout=5.0)
        data = response.json()
        
        # Status table
        table = Table(title="Bot Status", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", data.get("status", "Unknown"))
        table.add_row("Uptime", data.get("uptime", "N/A"))
        table.add_row("Total Trades", str(data.get("total_trades", 0)))
        table.add_row("Open Positions", str(data.get("open_positions", 0)))
        table.add_row("Total P&L", f"${data.get('total_pnl', 0):.2f}")
        table.add_row("Current Drawdown", f"{data.get('drawdown', 0):.1%}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Failed to connect to bot: {e}[/red]")
        console.print(f"[yellow]Make sure the bot is running on {host}:{port}[/yellow]")

if __name__ == "__main__":
    app()
