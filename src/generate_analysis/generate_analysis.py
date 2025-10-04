import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import re
import matplotlib.font_manager as fm


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




def apply_graph_styling(ax):
    """Apply consistent styling to graphs."""
    # Remove graph titles entirely
    ax.set_title("", fontsize=0)

    # Set font to Raleway globally
    raleway_path = fm.findfont(fm.FontProperties(family="Raleway"))
    if raleway_path:
        plt.rcParams["font.family"] = "Raleway"
    else:
        print("Raleway font not found. Using default font.")


def plot_avg_wins_per_pick(df: pd.DataFrame, out_path: str = "Output/avg_wins_per_pick.png") -> None:
    """Plot average number of wins per draft order as a bar plot."""
    avg_wins_per_pick = df.groupby("Pick")["W"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=avg_wins_per_pick, x="Pick", y="W", color="#4C72B0")
    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Bold and enlarge labels
    ax.set_xlabel("Draft Order (Pick Number)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Average Wins", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    apply_graph_styling(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
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
    # Bold and enlarge labels
    ax.set_xlabel("Pick", fontsize=16, fontweight="bold")
    ax.set_ylabel("Victorias", fontsize=16, fontweight="bold")
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
    apply_graph_styling(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
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
    ax.set_xlabel("Draft Order (Pick Number)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Average League Position", fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', labelsize=13)
    apply_graph_styling(ax)
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
    ax.set_xlabel("Pick", fontsize=16, fontweight="bold")
    ax.set_ylabel("Posición", fontsize=16, fontweight="bold")
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
    apply_graph_styling(ax)
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
    ax.set_xlabel("", fontsize=16, fontweight="bold")
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
    ax.set_xlabel("", fontsize=16, fontweight="bold")
    ax.set_ylabel("Victorias", fontsize=16, fontweight="bold")
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
        "total_wins",
        "total_losses",
        "win_percentage",
        "average_score",
        "highest_score",
        "lowest_score",
        "highest_position",
        "lowest_position",
    ]
    summary = summary[column_order]

    summary.to_csv(out_path, index=False)

# Ensure the function is defined before calling it
def plot_position_heatmap(df: pd.DataFrame, out_path: str = "Output/position_heatmap.png") -> None:
    """Generate a heatmap of positions by user and season."""
    # Pivot the data to create a matrix of Users (rows) x Seasons (columns)
    heatmap_data = df.pivot(index="User", columns="Season", values="Position")

    # Normalize positions per season for coloring (reverse shading: better positions darker)
    normalized_data = heatmap_data.rank(axis=0, method="min", ascending=True)  # Reverse the shading logic

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        normalized_data,
        annot=heatmap_data,
        fmt=".0f",
        cmap="Blues",  # Use the standard blue colormap for reversed logic
        linewidths=0.5,
        linecolor="lightgray",
        cbar=False,  # Remove the color legend
        mask=heatmap_data.isnull(),  # Black squares for missing data
    )

    # Customize annotations for text color
    for text in ax.texts:
        value = int(text.get_text()) if text.get_text().isdigit() else None
        if value is not None:
            if value <= 8:
                text.set_color("green")
                text.set_fontweight("bold")
            else:
                text.set_color("white")  # White for positions worse than 12
        
    # Set axis labels and title
    ax.set_title("Evolución año a año", fontsize=16, fontweight="bold")
    ax.set_xlabel("")  # Remove the default x-axis label
    ax.set_ylabel("", fontsize=14, fontweight="bold")

    # Adjust season labels to be on top and vertical
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    plt.xticks(rotation=90, ha="center", fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def clean_variable(name: str) -> str:
    """Clean variable name."""
    return re.sub(
        r"C\((User|Pick), Sum\)\[S\.([A-Za-z0-9_ ]+)\]",
        lambda m: m.group(2) if m.group(1) == "User" else f"pick_{m.group(2)}",
        name
    )

def postprocess_results_tables(results_df:pd.DataFrame) -> pd.DataFrame:
    """Postprocess regression results tables to clean variable names and round values."""
    # Clean variable names
    results_df["Variable"] = results_df["Variable"].apply(clean_variable)
    
    # Round numeric columns to one decimal place
    results_df["Coefficient"] = results_df["Coefficient"].round(1)
    results_df["P-Value"] = results_df["P-Value"].round(1)

    return results_df

def perform_linear_regression(df: pd.DataFrame, out_path: str = "Output/") -> None:
    """Perform linear regression with deviation coding and export results."""
    # Define the formula for deviation coding
    formula = "W ~ C(User, Sum) + C(Pick, Sum)"

    # Fit the model using statsmodels' formula API
    model = sm.OLS.from_formula(formula, data=df).fit()

    # Create a DataFrame for coefficients and significance
    results_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "P-Value": model.pvalues.values,
        "Significant": model.pvalues < 0.05
    })

    # Sort by coefficient (worst to best)
    results_df = results_df.sort_values(by="Coefficient")

    # Save the processed results
    clean_results = postprocess_results_tables(results_df)
    clean_results.to_csv(Path(out_path) / "regression_results.csv", index=False)

    # Filter for significant variables, sort by coefficient (best to worst), and save
    significant_results = clean_results[clean_results["Significant"]].sort_values(by="Coefficient", ascending=False)
    significant_results.to_csv(Path(out_path) / "significant_results.csv", index=False)


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
    plot_position_heatmap(df)  # New heatmap function
    perform_linear_regression(df)  # New regression analysis function


