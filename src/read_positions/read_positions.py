import pandas as pd
import pytesseract
from PIL import Image
import re

# Load image
img_path = "/mnt/data/24-25.png"
img = Image.open(img_path)

# OCR extraction
text = pytesseract.image_to_string(img)

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

print(df)
