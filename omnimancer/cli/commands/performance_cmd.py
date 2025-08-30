"""
Performance monitoring CLI command for Omnimancer.

Provides command-line access to performance metrics, dashboards, and optimization suggestions.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import click
from rich.console import Console
from rich.table import Table

from ...core.agent.metrics_collector import (
    get_metrics_collector,
    initialize_metrics_collection,
)
from ...core.agent.optimization_engine import (
    AlertLevel,
    OptimizationCategory,
    get_optimization_engine,
    initialize_optimization_engine,
)
from ...core.agent.performance_monitor import (
    get_performance_monitor,
    initialize_performance_monitoring,
)
from ...core.agent.token_tracker import get_token_tracker
from ..performance_dashboard import (
    create_performance_dashboard,
)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.pass_context
def performance(ctx, debug):
    """Performance monitoring and optimization commands."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    if debug:
        import logging

        logging.getLogger("omnimancer.core.agent").setLevel(logging.DEBUG)


@performance.command()
@click.option(
    "--start-monitoring",
    is_flag=True,
    help="Start performance monitoring services",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.pass_context
def status(ctx, start_monitoring, output_format):
    """Show performance monitoring status."""
    console = Console()

    try:
        if start_monitoring:
            console.print("[yellow]Initializing performance monitoring...[/yellow]")

            # Initialize all components
            performance_monitor = initialize_performance_monitoring(auto_start=True)
            metrics_collector = initialize_metrics_collection(auto_start=True)
            optimization_engine = initialize_optimization_engine(auto_start=True)

            console.print("[green]✓ Performance monitoring started[/green]")
        else:
            performance_monitor = get_performance_monitor()
            metrics_collector = get_metrics_collector()
            optimization_engine = get_optimization_engine()

        # Gather status information
        status_info = {
            "monitoring_active": (
                performance_monitor.is_monitoring
                if hasattr(performance_monitor, "is_monitoring")
                else False
            ),
            "metrics_collecting": (
                metrics_collector.is_collecting
                if hasattr(metrics_collector, "is_collecting")
                else False
            ),
            "optimization_running": (
                optimization_engine.is_running
                if hasattr(optimization_engine, "is_running")
                else False
            ),
            "timestamp": datetime.now().isoformat(),
        }

        # Get recent statistics
        dashboard_data = performance_monitor.get_performance_dashboard_data()
        status_info.update(
            {
                "active_alerts": len(dashboard_data.get("active_alerts", [])),
                "suggestions_count": len(dashboard_data.get("suggestions", [])),
                "recent_snapshot": dashboard_data.get("current_snapshot", {}),
            }
        )

        # Display results
        if output_format == "json":
            console.print(json.dumps(status_info, indent=2, default=str))
        else:
            _display_status_table(console, status_info)

    except Exception as e:
        console.print(f"[red]Error checking performance status: {e}[/red]")
        if ctx.obj.get("debug"):
            console.print_exception()
        sys.exit(1)


@performance.command()
@click.option(
    "--view",
    type=click.Choice(
        [
            "overview",
            "tokens",
            "performance",
            "resources",
            "alerts",
            "suggestions",
        ]
    ),
    default="overview",
    help="Dashboard view to display",
)
@click.option("--live", is_flag=True, help="Start live updating dashboard")
@click.option("--export", type=click.Path(), help="Export performance report to file")
@click.pass_context
def dashboard(ctx, view, live, export):
    """Show performance dashboard."""
    console = Console()

    try:
        dashboard = create_performance_dashboard()

        if export:
            export_path = Path(export)
            console.print(f"[yellow]Exporting performance report...[/yellow]")
            result_path = dashboard.export_report(export_path)
            console.print(f"[green]Report exported to {result_path}[/green]")
            return

        if live:
            console.print(f"[yellow]Starting live dashboard ({view} view)...[/yellow]")
            dashboard.start_live_dashboard(view)
        else:
            if view == "overview":
                dashboard.show_overview()
            else:
                dashboard.show_detailed_view(view)

    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error displaying dashboard: {e}[/red]")
        if ctx.obj.get("debug"):
            console.print_exception()
        sys.exit(1)


@performance.command()
@click.option("--hours", type=int, default=1, help="Time window in hours")
@click.option("--provider", help="Filter by specific provider")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.pass_context
def tokens(ctx, hours, provider, output_format):
    """Show token usage statistics."""
    console = Console()

    try:
        token_tracker = get_token_tracker()

        # Get usage summary
        time_window = timedelta(hours=hours)
        summary = token_tracker.get_usage_summary(time_window)

        if output_format == "json":
            console.print(json.dumps(summary, indent=2, default=str))
        else:
            _display_token_usage_table(console, summary, hours)

    except Exception as e:
        console.print(f"[red]Error retrieving token usage: {e}[/red]")
        if ctx.obj.get("debug"):
            console.print_exception()
        sys.exit(1)


@performance.command()
@click.option(
    "--category",
    type=click.Choice(
        [
            "cost_reduction",
            "performance_improvement",
            "resource_efficiency",
            "reliability_enhancement",
            "user_experience",
        ]
    ),
    help="Filter by optimization category",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.pass_context
def suggestions(ctx, category, output_format):
    """Show optimization suggestions."""
    console = Console()

    try:
        optimization_engine = get_optimization_engine()

        # Get current suggestions
        if category:
            category_enum = OptimizationCategory(category)
            current_suggestions = optimization_engine.get_current_suggestions(
                category_enum
            )
        else:
            current_suggestions = optimization_engine.get_current_suggestions()

        # Also get token tracker suggestions
        token_tracker = get_token_tracker()
        token_suggestions = token_tracker.get_cost_optimization_suggestions()

        if output_format == "json":
            data = {
                "optimization_suggestions": [s.__dict__ for s in current_suggestions],
                "token_suggestions": token_suggestions,
            }
            console.print(json.dumps(data, indent=2, default=str))
        else:
            _display_suggestions_table(console, current_suggestions, token_suggestions)

    except Exception as e:
        console.print(f"[red]Error retrieving suggestions: {e}[/red]")
        if ctx.obj.get("debug"):
            console.print_exception()
        sys.exit(1)


@performance.command()
@click.option(
    "--level",
    type=click.Choice(["info", "warning", "critical", "emergency"]),
    help="Filter by alert level",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.pass_context
def alerts(ctx, level, output_format):
    """Show performance alerts."""
    console = Console()

    try:
        optimization_engine = get_optimization_engine()

        # Get active alerts
        if level:
            level_enum = AlertLevel(level)
            active_alerts = optimization_engine.get_active_alerts(level_enum)
        else:
            active_alerts = optimization_engine.get_active_alerts()

        if output_format == "json":
            data = [alert.__dict__ for alert in active_alerts]
            console.print(json.dumps(data, indent=2, default=str))
        else:
            _display_alerts_table(console, active_alerts)

    except Exception as e:
        console.print(f"[red]Error retrieving alerts: {e}[/red]")
        if ctx.obj.get("debug"):
            console.print_exception()
        sys.exit(1)


@performance.command()
@click.option("--metric", help="Specific metric to analyze")
@click.option("--days", type=int, default=7, help="Analysis period in days")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.pass_context
def analyze(ctx, metric, days, output_format):
    """Analyze performance patterns and trends."""
    console = Console()

    try:
        token_tracker = get_token_tracker()
        metrics_collector = get_metrics_collector()

        # Analyze usage patterns
        usage_patterns = token_tracker.analyze_usage_patterns(days)

        analysis_results = {
            "usage_patterns": usage_patterns.__dict__,
            "analysis_period_days": days,
            "timestamp": datetime.now().isoformat(),
        }

        # Add metric-specific analysis if requested
        if metric:
            # Get anomalies for specific metric
            anomalies = metrics_collector.detect_anomalies(metric)
            analysis_results["anomalies"] = anomalies

        if output_format == "json":
            console.print(json.dumps(analysis_results, indent=2, default=str))
        else:
            _display_analysis_results(console, analysis_results)

    except Exception as e:
        console.print(f"[red]Error performing analysis: {e}[/red]")
        if ctx.obj.get("debug"):
            console.print_exception()
        sys.exit(1)


@performance.command()
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv"]),
    default="json",
    help="Export format",
)
@click.option("--days", type=int, default=7, help="Data period to export (days)")
@click.pass_context
def export(ctx, output, output_format, days):
    """Export performance metrics to file."""
    console = Console()

    try:
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"omnimancer_performance_{timestamp}.{output_format}"

        output_path = Path(output)
        time_window = timedelta(days=days)

        console.print(f"[yellow]Exporting {days} days of performance data...[/yellow]")

        # Export metrics
        metrics_collector = get_metrics_collector()
        metrics_path = metrics_collector.export_metrics(
            time_window, f"metrics_{output_path.stem}.json"
        )

        # Export token usage
        token_tracker = get_token_tracker()
        token_tracker.save_usage_history(f"tokens_{output_path.stem}.json")

        # Create combined report
        dashboard = create_performance_dashboard()
        report_path = dashboard.export_report(output_path)

        console.print(f"[green]✓ Performance data exported to {report_path}[/green]")
        console.print(f"[green]✓ Metrics data exported to {metrics_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error exporting data: {e}[/red]")
        if ctx.obj.get("debug"):
            console.print_exception()
        sys.exit(1)


# Helper functions for display
def _display_status_table(console: Console, status_info: Dict[str, Any]) -> None:
    """Display status information in table format."""
    table = Table(title="Performance Monitoring Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    # Monitoring status
    monitoring_status = (
        "🟢 Active" if status_info.get("monitoring_active") else "🔴 Inactive"
    )
    table.add_row("Performance Monitor", monitoring_status, "Real-time monitoring")

    # Metrics collection
    metrics_status = (
        "🟢 Active" if status_info.get("metrics_collecting") else "🔴 Inactive"
    )
    table.add_row("Metrics Collection", metrics_status, "System metrics gathering")

    # Optimization engine
    optimization_status = (
        "🟢 Running" if status_info.get("optimization_running") else "🔴 Stopped"
    )
    table.add_row(
        "Optimization Engine",
        optimization_status,
        "Alert and suggestion generation",
    )

    # Statistics
    alerts_count = status_info.get("active_alerts", 0)
    suggestions_count = status_info.get("suggestions_count", 0)

    table.add_row("Active Alerts", f"{alerts_count}", "Performance warnings")
    table.add_row("Suggestions", f"{suggestions_count}", "Optimization recommendations")

    console.print(table)


def _display_token_usage_table(
    console: Console, summary: Dict[str, Any], hours: int
) -> None:
    """Display token usage in table format."""
    table = Table(title=f"Token Usage (Last {hours} hours)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Rate", justify="right")

    total_tokens = summary.get("total_tokens", 0)
    total_requests = summary.get("total_requests", 0)
    total_cost = summary.get("total_cost", 0.0)

    table.add_row(
        "Total Tokens",
        f"{total_tokens:,}",
        f"{total_tokens/(hours*60):.0f}/min",
    )
    table.add_row(
        "Total Requests",
        f"{total_requests:,}",
        f"{total_requests/(hours*60):.1f}/min",
    )
    table.add_row(
        "Total Cost", f"${total_cost:.4f}", f"${total_cost/(hours*60):.5f}/min"
    )

    if total_requests > 0:
        table.add_row("Avg Tokens/Request", f"{total_tokens/total_requests:.0f}", "-")
        table.add_row("Avg Cost/Request", f"${total_cost/total_requests:.5f}", "-")

    console.print(table)

    # Provider breakdown
    provider_breakdown = summary.get("provider_breakdown", {})
    if provider_breakdown:
        provider_table = Table(title="Provider Breakdown")
        provider_table.add_column("Provider", style="cyan")
        provider_table.add_column("Tokens", justify="right")
        provider_table.add_column("Requests", justify="right")
        provider_table.add_column("Cost", justify="right")

        for provider, stats in provider_breakdown.items():
            provider_table.add_row(
                provider,
                f"{stats['tokens']:,}",
                f"{stats['requests']:,}",
                f"${stats['cost']:.4f}",
            )

        console.print()
        console.print(provider_table)


def _display_suggestions_table(
    console: Console,
    optimization_suggestions: List[Any],
    token_suggestions: List[Dict[str, Any]],
) -> None:
    """Display optimization suggestions in table format."""
    if not optimization_suggestions and not token_suggestions:
        console.print("[green]No optimization suggestions at this time[/green]")
        return

    table = Table(title="Optimization Suggestions")
    table.add_column("Type", style="cyan")
    table.add_column("Title")
    table.add_column("Impact")
    table.add_column("Difficulty")

    # Add optimization suggestions
    for suggestion in optimization_suggestions:
        table.add_row(
            (
                suggestion.optimization_type.value
                if hasattr(suggestion, "optimization_type")
                else "General"
            ),
            suggestion.title if hasattr(suggestion, "title") else "N/A",
            (
                suggestion.potential_improvement
                if hasattr(suggestion, "potential_improvement")
                else "Unknown"
            ),
            (
                suggestion.implementation_difficulty
                if hasattr(suggestion, "implementation_difficulty")
                else "Unknown"
            ),
        )

    # Add token suggestions
    for suggestion in token_suggestions:
        table.add_row(
            suggestion.get("type", "token"),
            suggestion.get("title", "Token Optimization"),
            suggestion.get("potential_savings", "Unknown"),
            "Easy",
        )

    console.print(table)


def _display_alerts_table(console: Console, alerts: List[Any]) -> None:
    """Display performance alerts in table format."""
    if not alerts:
        console.print("[green]No active alerts[/green]")
        return

    table = Table(title="Performance Alerts")
    table.add_column("Level", style="cyan")
    table.add_column("Title")
    table.add_column("Description")
    table.add_column("Time")

    for alert in alerts:
        level_color = {
            "info": "blue",
            "warning": "yellow",
            "critical": "red",
            "emergency": "bright_red",
        }.get(alert.level.value if hasattr(alert, "level") else "info", "white")

        table.add_row(
            f"[{level_color}]{alert.level.value if hasattr(alert, 'level') else 'Unknown'}[/{level_color}]",
            alert.title if hasattr(alert, "title") else "Unknown Alert",
            (alert.description if hasattr(alert, "description") else "No description"),
            (
                alert.timestamp.strftime("%H:%M:%S")
                if hasattr(alert, "timestamp")
                else "Unknown"
            ),
        )

    console.print(table)


def _display_analysis_results(console: Console, results: Dict[str, Any]) -> None:
    """Display analysis results in readable format."""
    usage_patterns = results.get("usage_patterns", {})

    # Summary table
    table = Table(
        title=f"Performance Analysis ({results.get('analysis_period_days', 7)} days)"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total Tokens", f"{usage_patterns.get('total_tokens', 0):,}")
    table.add_row("Total Requests", f"{usage_patterns.get('total_requests', 0):,}")
    table.add_row("Total Cost", f"${usage_patterns.get('total_cost', 0.0):.4f}")
    table.add_row(
        "Avg Tokens/Request",
        f"{usage_patterns.get('average_tokens_per_request', 0):.0f}",
    )
    table.add_row(
        "Peak Usage Time",
        str(usage_patterns.get("peak_usage_time", "Unknown")),
    )

    console.print(table)

    # Efficiency metrics
    if usage_patterns.get("average_context_utilization", 0) > 0:
        efficiency_table = Table(title="Efficiency Metrics")
        efficiency_table.add_column("Metric", style="cyan")
        efficiency_table.add_column("Score")

        efficiency_table.add_row(
            "Context Utilization",
            f"{usage_patterns.get('average_context_utilization', 0)*100:.1f}%",
        )
        efficiency_table.add_row(
            "Response Efficiency",
            f"{usage_patterns.get('average_response_efficiency', 0)*100:.1f}%",
        )

        console.print()
        console.print(efficiency_table)
