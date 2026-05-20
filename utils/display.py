from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from datetime import datetime

console = Console()

def print_header():
    console.print(Panel.fit(
        "[bold white]AGENTIC PROCTOR — LIVE MONITORING[/bold white]",
        box=box.DOUBLE,
        border_style="cyan"
    ))

def print_status(msg: str, style="cyan"):
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{ts}[/dim]  [{style}]{msg}[/{style}]")

def print_earphone_detections(phrases, logits):
    if not phrases:
        return
    table = Table(title="Earphone Scan Results", box=box.SIMPLE_HEAVY, border_style="yellow")
    table.add_column("Detection", style="white")
    table.add_column("Confidence", justify="right")
    for phrase, conf in zip(phrases, logits):
        conf_val = float(conf)
        color = "green" if conf_val > 0.5 else "yellow" if conf_val > 0.35 else "red"
        table.add_row(phrase, f"[{color}]{conf_val:.3f}[/{color}]")
    console.print(table)

def print_phone_detections(detections):
    if not detections:
        print_status("PHONE — no detections", "green")
        return
    table = Table(title="Phone Scan Results", box=box.SIMPLE_HEAVY, border_style="yellow")
    table.add_column("Label", style="white")
    table.add_column("Confidence", justify="right")
    table.add_column("BBox", justify="right")
    for det in detections:
        conf_val = det["confidence"]
        color = "green" if conf_val > 0.6 else "yellow" if conf_val > 0.4 else "red"
        table.add_row(det["label"], f"[{color}]{conf_val:.3f}[/{color}]", str(det["bbox"]))
    console.print(table)

def print_llm_result(result: dict, label="phone"):
    if not result:
        print_status(f"{label.upper()} LLM — no result", "red")
        return
    valid = result.get("valid", False)
    severity = result.get("severity", "LOW")
    reason = result.get("reason", "")
    color = "red" if valid else "green"
    sev_color = "red" if severity == "HIGH" else "yellow" if severity == "MEDIUM" else "blue"
    console.print(
        Panel(
            f"[dim]Valid:[/dim] [{color}]{valid}[/{color}]  "
            f"[dim]Severity:[/dim] [{sev_color}]{severity}[/{sev_color}]\n"
            f"[dim]Reason:[/dim] {reason}",
            title=f"{label.upper()} LLM RESULT",
            border_style=color,
            box=box.SIMPLE_HEAVY
        )
    )

def print_violation(violation: dict):
    sev = violation.get("severity", "LOW")
    sev_color = "red" if sev == "HIGH" else "yellow" if sev == "MEDIUM" else "blue"
    console.print(
        Panel(
            f"[dim]Type:[/dim]     {violation['label'].upper()}\n"
            f"[dim]Severity:[/dim] [{sev_color}]{sev}[/{sev_color}]\n"
            f"[dim]Reason:[/dim]   {violation['reason']}\n"
            f"[dim]Conf:[/dim]     {violation['confidence']}",
            title="[bold red]*** VIOLATION DETECTED ***[/bold red]",
            border_style="red",
            box=box.DOUBLE
        )
    )