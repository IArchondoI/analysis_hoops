import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_clean_picks() -> pd.DataFrame:
    """Load draft positions."""
    df = pd.read_csv(Path("Input/proc_inputs/picks.csv"))
    df.columns = ["Season", "Pick", "User"]
    df = df[df["User"] != "-"]
    df["User"] = df["User"].apply(lambda x: x.replace("marttinelli13", "jbena14"))
    df["User"] = df["User"].apply(lambda x: x.lower())
    df["User"] = df["User"].apply(lambda x: x.replace("antoniogrito", "antonio_grito").replace("andionrunbia", "andionrubia"))
    return df



def load_and_clean_positions() -> pd.DataFrame:
    """Load positions."""
    df = pd.read_csv(Path("Input/proc_inputs/positions.csv"))
    df = df[df["Season"] != "22-23_mal"]
    df["User"] = df["User"].apply(lambda x: x.lower())
    return df




def plot_avg_wins_per_pick(df: pd.DataFrame, out_path: str = "Output/avg_wins_per_pick.png") -> None:
    """Plot average number of wins per draft order as a bar plot."""
    avg_wins_per_pick = df.groupby("Pick")["W"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=avg_wins_per_pick, x="Pick", y="W", color="#4C72B0")
    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Bold and enlarge title and labels
    ax.set_title("Average Number of Wins per Draft Order", fontsize=20, fontweight="bold")
    ax.set_xlabel("Draft Order (Pick Number)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Average Wins", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_wins_vs_pick_scatter(df: pd.DataFrame, out_path: str = "Output/wins_vs_pick_scatter.png") -> None:
    """Plot scatter of wins vs draft order, colored by season."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Draw vertical dashed lines for each draft order
    unique_picks = sorted(df["Pick"].unique())
    for pick in unique_picks:
        ax.axvline(pick, color="#888", linestyle="--", linewidth=1, zorder=0)
    # Scatter plot without black border
    scatter = sns.scatterplot(data=df, x="Pick", y="W", hue="Season", palette="tab10", s=80, edgecolor=None, ax=ax)
    # Add average star for each draft order
    avg_wins_per_pick = df.groupby("Pick")["W"].mean().reset_index()
    ax.scatter(avg_wins_per_pick["Pick"], avg_wins_per_pick["W"], marker="*", s=250, color="gold", edgecolor="black", zorder=5, label="Average")
    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Set ticks for each draft order
    ax.set_xticks(unique_picks)
    # Bold and enlarge title and labels
    ax.set_title("Wins by Draft Order (Disaggregated by Season)", fontsize=20, fontweight="bold")
    ax.set_xlabel("Draft Order (Pick Number)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Wins", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate 'Average' if present
    seen = set()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, title="Season", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_avg_position_per_pick(df: pd.DataFrame, out_path: str = "Output/avg_position_per_pick.png") -> None:
    """Plot average league position per draft order as a bar plot."""
    avg_position_per_pick = df.groupby("Pick")["Position"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=avg_position_per_pick, x="Pick", y="Position", color="#4C72B0")
    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Bold and enlarge title and labels
    ax.set_title("Average League Position per Draft Order", fontsize=20, fontweight="bold")
    ax.set_xlabel("Draft Order (Pick Number)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Average League Position", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_position_vs_pick_scatter(df: pd.DataFrame, out_path: str = "Output/position_vs_pick_scatter.png") -> None:
    """Plot scatter of league position vs draft order, colored by season."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Shade area below 8.5 for playoff qualification with light green color
    ax.axhspan(0, 8.5, color="lightgreen", alpha=0.5, zorder=0, label="Playoff Zone")
    # Draw vertical dashed lines for each draft order
    unique_picks = sorted(df["Pick"].unique())
    for pick in unique_picks:
        ax.axvline(pick, color="#888", linestyle="--", linewidth=1, zorder=0)
    # Scatter plot without black border
    scatter = sns.scatterplot(data=df, x="Pick", y="Position", hue="Season", palette="tab10", s=80, edgecolor=None, ax=ax)
    # Add average star for each draft order (with dark outline)
    avg_position_per_pick = df.groupby("Pick")["Position"].mean().reset_index()
    ax.scatter(avg_position_per_pick["Pick"], avg_position_per_pick["Position"], marker="*", s=150, color="gold", edgecolor="black", zorder=5, label="Average")
    # Set ticks for each draft order and all positions
    ax.set_xticks(unique_picks)
    ax.set_yticks(range(1, int(df["Position"].max()) + 1))
    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Bold and enlarge title and labels
    ax.set_title("League Position by Draft Order (Disaggregated by Season)", fontsize=20, fontweight="bold")
    ax.set_xlabel("Draft Order (Pick Number)", fontsize=16, fontweight="bold")
    ax.set_ylabel("League Position", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate 'Average' if present
    seen = set()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, title="Season", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_avg_draft_position_per_user(df: pd.DataFrame, out_path: str = "Output/avg_draft_position_per_user.png") -> None:
    """Plot average draft position per user as a bar plot."""
    avg_draft_position_per_user = df.groupby("User")["Pick"].mean().reset_index()
    avg_draft_position_per_user = avg_draft_position_per_user.sort_values(by="Pick")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=avg_draft_position_per_user, x="User", y="Pick", color="#4C72B0")
    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Bold and enlarge title and labels
    ax.set_title("Average Draft Position per User", fontsize=20, fontweight="bold")
    ax.set_xlabel("User", fontsize=16, fontweight="bold")
    ax.set_ylabel("Average Draft Position", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_draft_positions_per_user(df: pd.DataFrame, out_path: str = "Output/draft_positions_per_user.png") -> None:
    """Plot draft positions per user, ordered by average draft position."""
    avg_draft_position_per_user = df.groupby("User")["Pick"].mean().reset_index()
    avg_draft_position_per_user = avg_draft_position_per_user.sort_values(by="Pick")
    user_order = avg_draft_position_per_user["User"].tolist()
    df["User"] = pd.Categorical(df["User"], categories=user_order, ordered=True)
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Draw vertical dashed lines for each user
    unique_users = user_order
    for user in unique_users:
        ax.axvline(user_order.index(user), color="#888", linestyle="--", linewidth=1, zorder=0)
    # Scatter plot without black border
    scatter = sns.stripplot(data=df, x="User", y="Pick", hue="Season", palette="tab10", size=8, jitter=True, ax=ax)
    # Add average star for each user (with dark outline)
    ax.scatter(avg_draft_position_per_user["User"], avg_draft_position_per_user["Pick"], marker="*", s=150, color="gold", edgecolor="black", zorder=5, label="Media")
    # Set ticks for each draft position
    ax.set_yticks(range(1, int(df["Pick"].max()) + 1))
    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Bold and enlarge title and labels
    ax.set_title("¿Es el bol injusto?", fontsize=20, fontweight="bold")
    ax.set_xlabel("Usuario", fontsize=16, fontweight="bold")
    ax.set_ylabel("Posición de Pickeo", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    plt.xticks(rotation=90, ha="center")
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate 'Average' if present
    seen = set()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, title="", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_avg_victories_per_user(df: pd.DataFrame, out_path: str = "Output/avg_victories_per_user.png") -> None:
    """Plot average number of victories per user as a bar plot."""
    avg_victories_per_user = df.groupby("User")["W"].mean().reset_index()
    avg_victories_per_user = avg_victories_per_user.sort_values(by="W", ascending=False)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=avg_victories_per_user, x="User", y="W", color="#4C72B0")
    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Bold and enlarge title and labels
    ax.set_title("Average Victories per User", fontsize=20, fontweight="bold")
    ax.set_xlabel("User", fontsize=16, fontweight="bold")
    ax.set_ylabel("Average Victories", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_victories_per_user_scatter(df: pd.DataFrame, out_path: str = "Output/victories_per_user_scatter.png") -> None:
    """Plot scatter of victories per user, with vertical lines and average stars."""
    avg_victories_per_user = df.groupby("User")["W"].mean().reset_index()
    avg_victories_per_user = avg_victories_per_user.sort_values(by="W", ascending=False)
    user_order = avg_victories_per_user["User"].tolist()
    df["User"] = pd.Categorical(df["User"], categories=user_order, ordered=True)
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Draw vertical dashed lines for each user
    for user in user_order:
        ax.axvline(user_order.index(user), color="#888", linestyle="--", linewidth=1, zorder=0)
    # Scatter plot without black border
    scatter = sns.stripplot(data=df, x="User", y="W", hue="Season", palette="tab10", size=8, jitter=True, ax=ax)
    # Add average star for each user (with dark outline)
    ax.scatter(avg_victories_per_user["User"], avg_victories_per_user["W"], marker="*", s=150, color="gold", edgecolor="black", zorder=5, label="Average")
    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Bold and enlarge title and labels
    ax.set_title("Victories per User (Disaggregated by Season)", fontsize=20, fontweight="bold")
    ax.set_xlabel("User", fontsize=16, fontweight="bold")
    ax.set_ylabel("Victories", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    plt.xticks(rotation=90, ha="center")
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate 'Average' if present
    seen = set()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, title="Season", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_user_summary_table(df: pd.DataFrame, out_path: str = "Output/user_summary_table.csv") -> None:
    """Generate a summary table for each user and export as CSV."""
    # Ensure numeric columns are properly cast
    df["W"] = pd.to_numeric(df["W"], errors="coerce")
    df["L"] = pd.to_numeric(df["L"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["Average"] = pd.to_numeric(df["Average"], errors="coerce")

    # Precompute win percentage for each row
    df["Win_Percentage"] = df["W"] / (df["W"] + df["L"]) * 100

    summary = (
        df.groupby("User", observed=False)  # Explicitly set observed=False to silence FutureWarning
        .agg(
            times_in_playoffs=("Position", lambda x: (x <= 8).sum()),
            times_played=("Season", "count"),
            total_wins=("W", "sum"),
            total_losses=("L", "sum"),
            win_percentage=("Win_Percentage", "mean"),
            average_score=("Average", "mean"),
            highest_score=("Average", "max"),
            lowest_score=("Average", "min"),
            highest_position=("Position", "min"),
            lowest_position=("Position", "max"),
        )
        .reset_index()
        .sort_values(by="win_percentage", ascending=False)  # Sort by Win Percentage
    )

    # Round win_percentage and average_score to 1 decimal place
    summary["win_percentage"] = summary["win_percentage"].round(1)
    summary["average_score"] = summary["average_score"].round(1)

    # Add total_games_played as the sum of total_wins and total_losses
    summary["total_games_played"] = summary["total_wins"] + summary["total_losses"]

    # Reorder columns to the desired order
    column_order = [
        "User",
        "times_in_playoffs",
        "times_played",
        "total_games_played",
        "win_percentage",
        "total_wins",
        "total_losses",
        "average_score",
        "highest_score",
        "lowest_score",
        "highest_position",
        "lowest_position",
    ]
    summary = summary[column_order]

    summary.to_csv(out_path, index=False)

# Update the main function to include the new table generation
def run_analysis_pipeline():
    """Load and execute all analyses."""
    picks = load_and_clean_picks()
    pos = load_and_clean_positions()
    # Merge picks and positions on Season and User
    df = pd.merge(picks, pos, on=["Season", "User"], how="inner")
    plot_avg_wins_per_pick(df)
    plot_wins_vs_pick_scatter(df)
    plot_avg_position_per_pick(df)
    plot_position_vs_pick_scatter(df)
    plot_avg_draft_position_per_user(picks)
    plot_draft_positions_per_user(picks)
    plot_avg_victories_per_user(df)
    plot_victories_per_user_scatter(df)
    generate_user_summary_table(df)
