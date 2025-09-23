
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


def main():
    picks = load_and_clean_picks()
    pos = load_and_clean_positions()
    # Merge picks and positions on Season and User
    df = pd.merge(picks, pos, on=["Season", "User"], how="inner")
    plot_avg_wins_per_pick(df)
    plot_wins_vs_pick_scatter(df)

# if __name__ == "__main__":
#     main()


picks = load_and_clean_picks()
pos = load_and_clean_positions()
# Merge picks and positions on Season and User
df = pd.merge(picks, pos, on=["Season", "User"], how="inner")
plot_avg_wins_per_pick(df)
plot_wins_vs_pick_scatter(df)



# Only one set of function definitions should exist above this line. The main function and main guard are already defined above.

