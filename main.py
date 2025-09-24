"""Main file to run whole pipeline."""
from src.read_positions.read_positions import extract_positions
from src.generate_analysis.generate_analysis import run_analysis_pipeline

from pathlib import Path

EXTRACT_POSITONS = False

def extract_positions_pipeline()->None:
    """Extracts positions from screenshots and saves them."""
    main_folder = Path("Input/poc_screenshots/")
    positions_table = extract_positions(main_folder)
    positions_table.to_csv("Input/proc_inputs/positions.csv",index=False)
    


def main() -> None:
    if EXTRACT_POSITONS:
        extract_positions_pipeline()
    run_analysis_pipeline()


if __name__ == "__main__":
    main()