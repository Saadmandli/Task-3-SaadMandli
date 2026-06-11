import pandas as pd
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.rule import Rule
from rich.prompt import Prompt

console = Console()


# ── banner ────────────────────────────────────────────────────────────────────

def show_banner():
    console.print()
    banner = Text(justify="center")
    banner.append("  ████████╗███████╗ ██████╗██╗  ██╗\n", style="bold cyan")
    banner.append("     ██╔══╝██╔════╝██╔════╝██║  ██║\n", style="bold cyan")
    banner.append("     ██║   █████╗  ██║     ███████║\n", style="bold cyan")
    banner.append("     ██║   ██╔══╝  ██║     ██╔══██║\n", style="bold cyan")
    banner.append("     ██║   ███████╗╚██████╗██║  ██║\n", style="bold cyan")
    banner.append("     ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝\n\n", style="bold cyan")
    banner.append("     TECH STACK RECOMMENDER SYSTEM\n", style="bold white")
    banner.append("     BUILT BY SAAD MANDLI", style="dim white")

    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))
    console.print()


# ── input section ─────────────────────────────────────────────────────────────

def take_inputs():
    console.print(Rule("[bold cyan]  STEP 1 — Enter Your Skills  ", style="cyan"))
    console.print("  [dim]Minimum 3 required for accurate matching[/dim]\n")

    skills = []

    for i in range(3):
        while True:
            val = Prompt.ask(f"  [cyan]Skill {i+1}[/cyan]").strip()
            if val:
                skills.append(val)
                console.print(f"    [green]✓[/green] [dim]{val}[/dim] added\n")
                break
            console.print("  [red]✗ Cannot be empty, try again[/red]")

    console.print("  [dim]Optional — press Enter to skip[/dim]")
    for i in range(3, 6):
        val = Prompt.ask(f"  [dim]Skill {i+1} (optional)[/dim]", default="").strip()
        if not val:
            break
        skills.append(val)
        console.print(f"    [green]✓[/green] [dim]{val}[/dim] added\n")

    # show entered skills as a neat tag row
    console.print()
    tags = "  ".join(f"[bold white on dark_cyan] {s} [/bold white on dark_cyan]" for s in skills)
    console.print(Panel(tags, title="[cyan]Your Profile[/cyan]", border_style="cyan", padding=(0, 2)))
    console.print()

    return skills


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    try:
        df = pd.read_csv("raw_skills.csv")
        df.columns = df.columns.str.strip()
        for col in df.columns:
            df[col] = df[col].str.strip()
        return df
    except FileNotFoundError:
        console.print("[bold red]  ERROR:[/bold red] raw_skills.csv not found — put it in the same folder")
        sys.exit()


# ── synonym map ───────────────────────────────────────────────────────────────

def build_syn_map(df):
    all_tokens = {}
    for _, row in df.iterrows():
        for t in row["skills"].split():
            all_tokens[t.lower()] = t

    def find_token(syn):
        if syn in all_tokens:
            return all_tokens[syn]
        syn_clean = syn.replace(" ", "").replace(".", "").replace("/", "").replace("-", "")
        for key, token in all_tokens.items():
            if syn_clean == key.replace(".", "").replace("/", "").replace("-", ""):
                return token
        return syn

    syn_map = {}
    for _, row in df.iterrows():
        for syn in str(row["synonyms"]).split("|"):
            s = syn.strip().lower()
            if s and s not in syn_map:
                syn_map[s] = find_token(s)

    return syn_map


def normalize(skill, syn_map):
    return syn_map.get(skill.lower().strip(), skill.strip())


# ── engine ────────────────────────────────────────────────────────────────────

def recommend(df, user_skills, syn_map):
    normalized = [normalize(s, syn_map) for s in user_skills]
    user_str = " ".join(normalized)

    vec = TfidfVectorizer()
    skill_matrix = vec.fit_transform(df["skills"])
    user_vec = vec.transform([user_str])

    scores = cosine_similarity(user_vec, skill_matrix)[0]
    df = df.copy()
    df["score"] = scores
    return df.sort_values("score", ascending=False)


# ── processing animation ──────────────────────────────────────────────────────

def run_with_progress(df, user_skills, syn_map):
    console.print(Rule("[bold cyan]  STEP 2 — Processing  ", style="cyan"))
    console.print()

    steps = [
        ("Loading job role dataset",       0.8),
        ("Building synonym map from CSV",  0.6),
        ("Applying TF-IDF vectorization",  1.0),
        ("Running cosine similarity",       1.0),
        ("Ranking and filtering results",  0.6),
    ]

    with Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=28, complete_style="cyan", finished_style="green"),
        TextColumn("[green]{task.percentage:>3.0f}%"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Starting...", total=len(steps))
        for desc, secs in steps:
            progress.update(task, description=f"  {desc}")
            time.sleep(secs)
            progress.advance(task)

    console.print()
    return recommend(df, user_skills, syn_map)


# ── results ───────────────────────────────────────────────────────────────────

def bar_colored(pct):
    filled = int(pct / 5)
    empty  = 20 - filled
    if pct >= 50:
        color = "green"
    elif pct >= 25:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"


def show_all_scores(ranked):
    console.print(Rule("[bold cyan]  STEP 3 — All Roles Scored  ", style="cyan"))
    console.print()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        border_style="dim cyan",
        padding=(0, 1),
        expand=False,
    )
    table.add_column("Job Role",       style="white",      min_width=28)
    table.add_column("Match Score",    style="bold white", min_width=8,  justify="right")
    table.add_column("Visual",         min_width=22)
    table.add_column("Rank",           style="dim white",  min_width=4,  justify="center")

    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        pct = round(row["score"] * 100, 1)
        table.add_row(
            row["job_role"],
            f"{pct}%",
            Text.from_markup(bar_colored(pct)),
            f"#{rank}",
        )

    console.print(table)
    console.print()


def show_top3(top3, user_skills):
    console.print(Rule("[bold cyan]  STEP 4 — Top Matches For You  ", style="cyan"))
    console.print()

    medals   = ["🥇", "🥈", "🥉"]
    colors   = ["gold1", "grey74", "orange3"]
    labels   = ["1ST PLACE", "2ND PLACE", "3RD PLACE"]

    cards = []
    for i, (_, row) in enumerate(top3.iterrows()):
        pct     = round(row["score"] * 100, 1)
        preview = row["skills"][:55] + "..." if len(row["skills"]) > 55 else row["skills"]

        content = Text(justify="left")
        content.append(f"{medals[i]}  {row['job_role']}\n", style=f"bold {colors[i]}")
        content.append("─" * 32 + "\n",                     style="dim")
        content.append(f"  Match   :  ",                    style="dim white")
        content.append(f"{pct}%\n",                         style=f"bold {colors[i]}")
        content.append(Text.from_markup(f"  Score   :  {bar_colored(pct)}\n"))
        content.append(f"  Skills  :  ",                    style="dim white")
        content.append(f"{preview}\n",                       style="dim cyan")

        cards.append(Panel(
            content,
            title=f"[bold {colors[i]}]{labels[i]}[/bold {colors[i]}]",
            border_style=colors[i],
            padding=(1, 2),
            width=46,
        ))

    console.print(Columns(cards, equal=True, expand=False))
    console.print()

    if top3["score"].max() < 0.05:
        console.print(Panel(
            "[yellow]⚠  No strong match found for your skill set.\n"
            "   Try: Data Scientist | DevOps Engineer | Full Stack Developer[/yellow]",
            border_style="yellow",
        ))
        console.print()


def show_summary(top3, user_skills):
    console.print(Rule("[bold cyan]  Summary  ", style="cyan"))
    console.print()

    # skills row
    tags = "  ".join(f"[bold white on dark_cyan] {s} [/bold white on dark_cyan]" for s in user_skills)
    console.print(f"  [dim]Skills analyzed:[/dim]  {tags}")
    console.print()

    # summary table
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
        padding=(0, 2),
    )
    table.add_column("Rank",      justify="center", style="bold cyan",  min_width=6)
    table.add_column("Job Role",  style="bold white", min_width=28)
    table.add_column("Match %",   justify="right",  style="bold green", min_width=8)
    table.add_column("Verdict",   style="dim white", min_width=16)

    verdicts = ["Strong Match ✓", "Good Match ✓", "Possible Match"]

    for i, (_, row) in enumerate(top3.iterrows()):
        pct = round(row["score"] * 100, 1)
        table.add_row(f"#{i+1}", row["job_role"], f"{pct}%", verdicts[i])

    console.print(table)
    console.print()

    # footer
    console.print(Panel(
        "[dim]  Built by [bold white]Saad Mandli[/bold white]  ·  "
        "Algorithm: TF-IDF + Cosine Similarity[/dim]",
        border_style="dim cyan",
        padding=(0, 2),
    ))
    console.print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    show_banner()

    user_skills = take_inputs()

    df      = load_data()
    syn_map = build_syn_map(df)
    ranked  = run_with_progress(df, user_skills, syn_map)

    show_all_scores(ranked)
    show_top3(ranked.head(3), user_skills)
    show_summary(ranked.head(3), user_skills)


if __name__ == "__main__":
    main()