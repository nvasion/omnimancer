"""
Performance Dashboard for Omnimancer Agent Monitoring.

This module provides a rich CLI interface for viewing agent performance metrics,
token usage, resource utilization, and optimization suggestions.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..core.agent.metrics_collector import (
    get_metrics_collector,
)
from ..core.agent.performance_monitor import (
    get_performance_monitor,
)
from ..core.agent.token_tracker import (
    get_token_tracker,
)

logger = logging.getLogger(__name__)


class PerformanceDashboard:
    """Interactive performance dashboard for Omnimancer agents."""

    def __init__(self):
        """Initialize the dashboard."""
        self.console = Console()
        self.performance_monitor = get_performance_monitor()
        self.token_tracker = get_token_tracker()
        self.metrics_collector = get_metrics_collector()

        # Dashboard state
        self.is_running = False
        self.refresh_interval = 5.0  # seconds
        self.current_view = "overview"

        # Views
        self.available_views = {
            "overview": self._render_overview,
            "tokens": self._render_token_usage,
            "performance": self._render_performance_metrics,
            "resources": self._render_resource_usage,
            "alerts": self._render_alerts,
            "suggestions": self._render_suggestions,
        }

    def show_overview(self) -> None:
        """Show overview dashboard (non-interactive)."""
        try:
            self.console.clear()
            self._print_header("Omnimancer Performance Overview")

            # Get current data
            dashboard_data = self.performance_monitor.get_performance_dashboard_data()
            metrics_data = self.metrics_collector.get_dashboard_metrics()
            token_summary = self.token_tracker.get_usage_summary(timedelta(hours=1))

            # Create layout
            layout = self._create_overview_layout(
                dashboard_data, metrics_data, token_summary
            )
            self.console.print(layout)

            # Show recent alerts
            recent_alerts = dashboard_data.get("active_alerts", [])[
                -3:
            ]  # Last 3 alerts
            if recent_alerts:
                self.console.print(Rule("Recent Alerts"))
                self._display_alerts_summary(recent_alerts)

            # Show suggestions
            suggestions = dashboard_data.get("suggestions", [])[
                -2:
            ]  # Last 2 suggestions
            if suggestions:
                self.console.print(Rule("Optimization Suggestions"))
                self._display_suggestions_summary(suggestions)

        except Exception as e:
            self.console.print(f"[red]Error displaying overview: {e}[/red]")
            logger.error(f"Dashboard overview error: {e}")

    def show_detailed_view(self, view_name: str = "overview") -> None:
        """Show detailed view for specific metrics."""
        if view_name not in self.available_views:
            self.console.print(f"[red]Unknown view: {view_name}[/red]")
            self.console.print(
                f"Available views: {', '.join(self.available_views.keys())}"
            )
            return

        try:
            self.console.clear()
            self.available_views[view_name]()
        except Exception as e:
            self.console.print(f"[red]Error displaying {view_name} view: {e}[/red]")
            logger.error(f"Dashboard view error: {e}")

    def start_live_dashboard(self, view: str = "overview") -> None:
        """Start live updating dashboard."""
        if view not in self.available_views:
            self.console.print(f"[red]Unknown view: {view}[/red]")
            return

        self.current_view = view
        self.is_running = True

        try:
            with Live(
                self._generate_live_layout(),
                console=self.console,
                refresh_per_second=1 / self.refresh_interval,
                screen=True,
            ) as live:
                self.console.print(
                    "[yellow]Live dashboard started. Press Ctrl+C to exit.[/yellow]"
                )

                while self.is_running:
                    try:
                        live.update(self._generate_live_layout())
                        time.sleep(self.refresh_interval)
                    except KeyboardInterrupt:
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self.is_running = False
            self.console.print("\n[yellow]Dashboard stopped.[/yellow]")

    def _generate_live_layout(self) -> Layout:
        """Generate live updating layout."""
        if self.current_view == "overview":
            dashboard_data = self.performance_monitor.get_performance_dashboard_data()
            metrics_data = self.metrics_collector.get_dashboard_metrics()
            token_summary = self.token_tracker.get_usage_summary(timedelta(hours=1))
            return self._create_overview_layout(
                dashboard_data, metrics_data, token_summary
            )
        else:
            # For other views, just render statically
            layout = Layout()
            layout.split_column(
                Layout(self._render_header(), size=3),
                Layout(
                    Panel(
                        self.available_views[self.current_view](),
                        title=f"{self.current_view.title()} View",
                    )
                ),
            )
            return layout

    def _create_overview_layout(
        self, dashboard_data: Dict, metrics_data: Dict, token_summary: Dict
    ) -> Layout:
        """Create overview dashboard layout."""
        # Main layout
        layout = Layout()

        # Split into header and body
        layout.split_column(Layout(self._render_header(), size=3), Layout(name="body"))

        # Split body into sections
        layout["body"].split_row(Layout(name="left"), Layout(name="right"))

        # Left column: performance and resources
        layout["left"].split_column(
            Layout(self._create_performance_panel(dashboard_data, metrics_data)),
            Layout(self._create_resources_panel(metrics_data)),
        )

        # Right column: tokens and status
        layout["right"].split_column(
            Layout(self._create_token_usage_panel(token_summary)),
            Layout(self._create_status_panel(dashboard_data)),
        )

        return layout

    def _render_header(self) -> Panel:
        """Render dashboard header."""
        title_text = Text("Omnimancer Agent Performance Dashboard", style="bold blue")
        subtitle_text = Text(
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style="dim",
        )
        header_content = Align.center(f"{title_text}\n{subtitle_text}")

        return Panel(header_content, style="blue", box=box.ROUNDED)

    def _create_performance_panel(
        self, dashboard_data: Dict, metrics_data: Dict
    ) -> Panel:
        """Create performance metrics panel."""
        current_snapshot = dashboard_data.get("current_snapshot", {})
        performance_metrics = metrics_data.get("performance", {})

        # Performance table
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        # Response time
        response_time = performance_metrics.get("avg_response_time_ms", 0)
        response_status = self._get_status_indicator(
            response_time, 1000, 5000
        )  # Good < 1s, Warning < 5s
        table.add_row("Avg Response Time", f"{response_time:.0f} ms", response_status)

        # Success rate
        success_rate = performance_metrics.get("success_rate_percent", 100)
        success_status = self._get_status_indicator(
            success_rate, 95, 90, reverse=True
        )  # Good > 95%
        table.add_row("Success Rate", f"{success_rate:.1f}%", success_status)

        # Active operations
        active_ops = current_snapshot.get("active_operations", 0)
        ops_status = self._get_status_indicator(
            active_ops, 5, 15
        )  # Good < 5, Warning < 15
        table.add_row("Active Operations", str(active_ops), ops_status)

        return Panel(table, title="Performance Metrics", border_style="green")

    def _create_resources_panel(self, metrics_data: Dict) -> Panel:
        """Create system resources panel."""
        resources = metrics_data.get("resources", {})

        # Resources table
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", justify="right")
        table.add_column("Status", justify="center")

        # CPU usage
        cpu_percent = resources.get("cpu_percent", 0)
        cpu_status = self._get_status_indicator(
            cpu_percent, 50, 80
        )  # Good < 50%, Warning < 80%
        table.add_row("CPU Usage", f"{cpu_percent:.1f}%", cpu_status)

        # Memory usage
        memory_mb = resources.get("memory_mb", 0)
        memory_gb = memory_mb / 1024
        memory_status = self._get_status_indicator(
            memory_mb, 512, 1024
        )  # Good < 512MB, Warning < 1GB
        table.add_row("Memory Usage", f"{memory_gb:.1f} GB", memory_status)

        return Panel(table, title="System Resources", border_style="yellow")

    def _create_token_usage_panel(self, token_summary: Dict) -> Panel:
        """Create token usage panel."""
        # Token usage table
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Last Hour", justify="right")
        table.add_column("Rate", justify="right")

        # Token totals
        total_tokens = token_summary.get("total_tokens", 0)
        total_requests = token_summary.get("total_requests", 0)
        avg_tokens = token_summary.get("average_tokens_per_request", 0)

        table.add_row("Total Tokens", f"{total_tokens:,}", f"{total_tokens/60:.0f}/min")
        table.add_row("Requests", f"{total_requests:,}", f"{total_requests/60:.1f}/min")
        table.add_row("Avg Tokens/Request", f"{avg_tokens:.0f}", "-")

        # Cost information
        total_cost = token_summary.get("total_cost", 0.0)
        avg_cost = token_summary.get("average_cost_per_request", 0.0)

        table.add_row("Total Cost", f"${total_cost:.3f}", f"${total_cost/60:.4f}/min")
        table.add_row("Avg Cost/Request", f"${avg_cost:.4f}", "-")

        return Panel(table, title="Token Usage (Last Hour)", border_style="blue")

    def _create_status_panel(self, dashboard_data: Dict) -> Panel:
        """Create monitoring status panel."""
        monitoring_status = dashboard_data.get("monitoring_status", "unknown")
        active_alerts = len(dashboard_data.get("active_alerts", []))
        suggestions = len(dashboard_data.get("suggestions", []))

        # Status information
        status_lines = [
            f"Monitoring: {monitoring_status.title()}",
            f"Active Alerts: {active_alerts}",
            f"Suggestions: {suggestions}",
            "",
            f"Uptime: {self._get_uptime()}",
        ]

        status_text = "\n".join(status_lines)
        status_color = "green" if monitoring_status == "active" else "red"

        return Panel(status_text, title="System Status", border_style=status_color)

    def _get_status_indicator(
        self,
        value: float,
        good_threshold: float,
        warning_threshold: float,
        reverse: bool = False,
    ) -> str:
        """Get colored status indicator based on thresholds."""
        if reverse:
            # Higher values are better (e.g., success rate)
            if value >= good_threshold:
                return "[green]●[/green]"
            elif value >= warning_threshold:
                return "[yellow]●[/yellow]"
            else:
                return "[red]●[/red]"
        else:
            # Lower values are better (e.g., response time)
            if value <= good_threshold:
                return "[green]●[/green]"
            elif value <= warning_threshold:
                return "[yellow]●[/yellow]"
            else:
                return "[red]●[/red]"

    def _get_uptime(self) -> str:
        """Get system uptime string."""
        # This is a placeholder - in reality you'd track when monitoring started
        return "Running"

    def _print_header(self, title: str) -> None:
        """Print dashboard header."""
        self.console.print(Panel(title, style="bold blue"))
        self.console.print()

    def _render_overview(self) -> None:
        """Render overview page."""
        self._print_header("Performance Overview")

        # Get data and display
        dashboard_data = self.performance_monitor.get_performance_dashboard_data()
        self.console.print(json.dumps(dashboard_data, indent=2, default=str))

    def _render_token_usage(self) -> None:
        """Render token usage page."""
        self._print_header("Token Usage Analysis")

        # Recent usage summary
        summary_1h = self.token_tracker.get_usage_summary(timedelta(hours=1))
        summary_24h = self.token_tracker.get_usage_summary(timedelta(hours=24))

        # Create comparison table
        table = Table(
            title="Token Usage Summary",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Last Hour", justify="right")
        table.add_column("Last 24 Hours", justify="right")

        table.add_row(
            "Total Tokens",
            f"{summary_1h.get('total_tokens', 0):,}",
            f"{summary_24h.get('total_tokens', 0):,}",
        )
        table.add_row(
            "Total Requests",
            f"{summary_1h.get('total_requests', 0):,}",
            f"{summary_24h.get('total_requests', 0):,}",
        )
        table.add_row(
            "Total Cost",
            f"${summary_1h.get('total_cost', 0):.3f}",
            f"${summary_24h.get('total_cost', 0):.3f}",
        )

        self.console.print(table)
        self.console.print()

        # Provider breakdown
        if summary_24h.get("provider_breakdown"):
            provider_table = Table(
                title="Provider Breakdown (24h)",
                show_header=True,
                header_style="bold magenta",
            )
            provider_table.add_column("Provider", style="cyan")
            provider_table.add_column("Tokens", justify="right")
            provider_table.add_column("Requests", justify="right")
            provider_table.add_column("Cost", justify="right")

            for provider, stats in summary_24h["provider_breakdown"].items():
                provider_table.add_row(
                    provider,
                    f"{stats['tokens']:,}",
                    f"{stats['requests']:,}",
                    f"${stats['cost']:.3f}",
                )

            self.console.print(provider_table)

    def _render_performance_metrics(self) -> None:
        """Render performance metrics page."""
        self._print_header("Performance Metrics")

        metrics_data = self.metrics_collector.get_dashboard_metrics()

        # Performance summary
        performance = metrics_data.get("performance", {})

        perf_table = Table(
            title="Performance Summary",
            show_header=True,
            header_style="bold magenta",
        )
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", justify="right")
        perf_table.add_column("Status", justify="center")

        response_time = performance.get("avg_response_time_ms", 0)
        success_rate = performance.get("success_rate_percent", 100)

        perf_table.add_row(
            "Average Response Time",
            f"{response_time:.0f} ms",
            self._get_status_indicator(response_time, 1000, 5000),
        )
        perf_table.add_row(
            "Success Rate",
            f"{success_rate:.1f}%",
            self._get_status_indicator(success_rate, 95, 90, reverse=True),
        )

        self.console.print(perf_table)

    def _render_resource_usage(self) -> None:
        """Render resource usage page."""
        self._print_header("System Resource Usage")

        metrics_data = self.metrics_collector.get_dashboard_metrics()
        resources = metrics_data.get("resources", {})

        resource_table = Table(
            title="Current Resource Usage",
            show_header=True,
            header_style="bold magenta",
        )
        resource_table.add_column("Resource", style="cyan")
        resource_table.add_column("Current Usage", justify="right")
        resource_table.add_column("Status", justify="center")

        cpu_percent = resources.get("cpu_percent", 0)
        memory_mb = resources.get("memory_mb", 0)

        resource_table.add_row(
            "CPU",
            f"{cpu_percent:.1f}%",
            self._get_status_indicator(cpu_percent, 50, 80),
        )
        resource_table.add_row(
            "Memory",
            f"{memory_mb:.0f} MB",
            self._get_status_indicator(memory_mb, 512, 1024),
        )

        self.console.print(resource_table)

    def _render_alerts(self) -> None:
        """Render alerts page."""
        self._print_header("Performance Alerts")

        dashboard_data = self.performance_monitor.get_performance_dashboard_data()
        alerts = dashboard_data.get("active_alerts", [])

        if not alerts:
            self.console.print("[green]No active alerts[/green]")
            return

        for alert in alerts[-10:]:  # Show last 10 alerts
            self._display_alert_detail(alert)

    def _render_suggestions(self) -> None:
        """Render optimization suggestions page."""
        self._print_header("Optimization Suggestions")

        dashboard_data = self.performance_monitor.get_performance_dashboard_data()
        suggestions = dashboard_data.get("suggestions", [])

        # Also get token usage suggestions
        token_suggestions = self.token_tracker.get_cost_optimization_suggestions()

        all_suggestions = suggestions + token_suggestions

        if not all_suggestions:
            self.console.print("[green]No optimization suggestions available[/green]")
            return

        for suggestion in all_suggestions:
            self._display_suggestion_detail(suggestion)

    def _display_alerts_summary(self, alerts: List[Dict]) -> None:
        """Display alerts summary."""
        for alert in alerts:
            level_color = {
                "info": "blue",
                "warning": "yellow",
                "critical": "red",
                "emergency": "bright_red",
            }.get(alert.get("level", "info"), "white")

            self.console.print(
                f"[{level_color}]● {alert.get('title', 'Unknown Alert')}[/{level_color}]"
            )

    def _display_suggestions_summary(self, suggestions: List[Dict]) -> None:
        """Display suggestions summary."""
        for suggestion in suggestions:
            self.console.print(
                f"[yellow]💡 {suggestion.get('title', 'Optimization Available')}[/yellow]"
            )

    def _display_alert_detail(self, alert: Dict) -> None:
        """Display detailed alert information."""
        level_color = {
            "info": "blue",
            "warning": "yellow",
            "critical": "red",
            "emergency": "bright_red",
        }.get(alert.get("level", "info"), "white")

        alert_panel = Panel(
            f"[bold]{alert.get('description', 'No description')}[/bold]\n\n"
            + f"Threshold: {alert.get('threshold_value', 'N/A')}\n"
            + f"Current: {alert.get('current_value', 'N/A')}\n"
            + f"Time: {alert.get('timestamp', 'Unknown')}",
            title=f"[{level_color}]{alert.get('title', 'Alert')}[/{level_color}]",
            border_style=level_color,
        )

        self.console.print(alert_panel)
        self.console.print()

    def _display_suggestion_detail(self, suggestion: Dict) -> None:
        """Display detailed suggestion information."""
        suggestion_panel = Panel(
            f"[bold]{suggestion.get('description', 'No description')}[/bold]\n\n"
            + f"Suggestion: {suggestion.get('suggestion', 'N/A')}\n"
            + f"Potential Savings: {suggestion.get('potential_savings', 'Unknown')}",
            title=f"[yellow]💡 {suggestion.get('title', 'Optimization Suggestion')}[/yellow]",
            border_style="yellow",
        )

        self.console.print(suggestion_panel)
        self.console.print()

    def export_report(self, output_path: Optional[Path] = None) -> Path:
        """Export performance report to file."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"performance_report_{timestamp}.json")

        # Collect all data
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "performance_dashboard",
                "version": "1.0",
            },
            "dashboard_data": self.performance_monitor.get_performance_dashboard_data(),
            "metrics_data": self.metrics_collector.get_dashboard_metrics(),
            "token_summary_1h": self.token_tracker.get_usage_summary(
                timedelta(hours=1)
            ),
            "token_summary_24h": self.token_tracker.get_usage_summary(
                timedelta(hours=24)
            ),
            "usage_patterns": asdict(self.token_tracker.analyze_usage_patterns(7)),
            "optimization_suggestions": self.token_tracker.get_cost_optimization_suggestions(),
        }

        # Write report
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        self.console.print(
            f"[green]Performance report exported to {output_path}[/green]"
        )
        return output_path


def create_performance_dashboard() -> PerformanceDashboard:
    """Create a performance dashboard instance."""
    return PerformanceDashboard()
