import pandas as pd
import pytesseract
from PIL import Image
import re
from pathlib import Path
import cv2

def get_files_from_folder(input_path:Path)->list[Path]:
    """Get all .png files in a given folder, sorted alphabetically."""
    return sorted([f for f in input_path.glob('*.png') if f.is_file()])

def preprocess_image(img_path:Path):
    """Preprocess image."""
    # Load with OpenCV
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: dilate to make text bolder
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    # thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Increase contrast & threshold
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def process_image(img_path:Path)->pd.DataFrame:
    """Process a given image to extract results."""

    preprocessed_image = preprocess_image(img_path)

    # OCR extraction
    custom_oem_psm_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_image,config=custom_oem_psm_config)

    # Split lines, clean, and keep rows that look like data
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Prepare list for dataframe rows
    rows = []
    position = 1

    for line in lines:
        # Match pattern like: Team (User) Record Average Streak
        match = re.match(r"(.+?)\s\((.+?)\)\s+(\d+)-(\d+)\s+([\d\.]+)", line)
        if match:
            team, user, w, l, avg = match.groups()
            rows.append([position, team.strip(), user.strip(), int(w), int(l), float(avg)])
            position += 1

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["Position", "Team", "User", "W", "L", "Average"])
    df["Season"] = str(img_path).split("/")[-1].replace(".png","")
    return df

def extract_positions(main_folder:Path)->pd.DataFrame:
    """Extract positions from all tables and concatenate."""
    table_files = get_files_from_folder(main_folder)
    return pd.concat([extract_positions(image) for image in table_files])


main_folder = Path("Input/poc_screenshots/")

# extract_positions(main_folder)

table_files = get_files_from_folder(main_folder)

process_image(table_files[1])