
import pandas as pd
import pytesseract
import re
from pathlib import Path
import cv2


def get_files_from_folder(input_path: Path) -> list[Path]:
    """Get all .png files in a given folder, sorted alphabetically."""
    return sorted([f for f in input_path.glob("*.png") if f.is_file()])


def preprocess_image_with_cv2(img_path: Path):
    """Preprocess image."""
    # Load with OpenCV
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {img_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: dilate to make text bolder
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.dilate(gray, kernel, iterations=1)

    # Increase contrast & threshold
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]



def process_image_with_pytesseract(image_path: Path) -> pd.DataFrame:
    """Process image with pytesseract."""
    preprocessed_image = preprocess_image_with_cv2(image_path)

    # OCR extraction
    custom_config = r"--oem 3 --psm 6"
    ocr_data = pytesseract.image_to_data(
        preprocessed_image,
        config=custom_config,
        output_type=pytesseract.Output.DATAFRAME,
    )

    # Clean
    ocr_data = ocr_data[ocr_data["text"].notna()].reset_index(drop=True)

    # Group by line number
    lines = (
        ocr_data.groupby("line_num")["text"].apply(lambda x: " ".join(x)).tolist()
    )

    # Parse lines into structured rows
    rows = []
    position = 1
    for line in lines:
        if not line.startswith("Team"):
            line = (
                line.replace("{", "(")
                .replace("}", ")")
                .replace("((", "(")
                .replace("))", ")")
                .strip()
            )
            if "." in line:
                cut_idx = line.rfind(".")
                # include one digit after the dot at least
                line = line[: cut_idx + 2]
            else:
                raise ValueError("No dot")
            # Extract user
            user_match = re.search(r"\(([^)]+)\)", line)
            user = user_match.group(1).strip() if user_match else ""

            # --- Team ---
            if "(" not in line:
                raise ValueError("No user in line")
            team = line.split("(")[0].strip() if "(" in line else line

            # --- Record ---
            rec_match = re.search(r"(\d{1,3})\s*-\s*(\d{1,3})", line)
            if not rec_match:
                raise ValueError("No record in line")
            w, l = int(rec_match.group(1)), int(rec_match.group(2))

            # --- Average ---
            avg_match = re.search(r"(\d{1,3}(?:[.,]\d+)?)(?!.*\d)", line)
            if not avg_match:
                raise ValueError("No average in line")
            avg = float(avg_match.group(1).replace(",", "."))

            # Append row
            rows.append([position, team, user, w, l, avg])
            position += 1

    df = pd.DataFrame(
        rows, columns=["Position", "Team", "User", "W", "L", "Average"]
    )
    df["Season"] = str(image_path).split("/")[-1].replace(".png", "")
    return df


def clean_table(final_table: pd.DataFrame) -> pd.DataFrame:
    """Clean table."""
    final_table["User"] = final_table["User"].apply(
        lambda x: "IArchondo" if "archondo" in x.lower() else x
    )
    final_table["User"] = final_table["User"].apply(
        lambda x: x.replace(" +1", "").strip()
    )
    final_table["User"] = final_table["User"].apply(
        lambda x: x.replace("villugo", "villuqo")
        .replace("Perezlll", "PerezIII")
        .replace("marttinelli13", "jbena14")
        .replace("A_Andres", "A_andres")
        .strip()
    )
    return final_table


def extract_positions(main_folder: Path) -> pd.DataFrame:
    """Extract positions from all tables and concatenate."""
    table_files = get_files_from_folder(main_folder)
    final_table = pd.concat(
        [process_image_with_pytesseract(image) for image in table_files]
    )
    return clean_table(final_table)


main_folder = Path("Input/poc_screenshots/")
out = extract_positions(main_folder)
