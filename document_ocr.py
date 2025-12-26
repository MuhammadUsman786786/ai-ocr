# document_ocr.py
import cv2
import pytesseract
import numpy as np
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
TABLE_LINE_THRESHOLD = 5000   # tune once if needed
MIN_CELL_WIDTH = 80
MIN_CELL_HEIGHT = 25
TABLE_COLUMNS = 3             # Description | Specified | Offered

# -----------------------------
# TABLE DETECTION
# -----------------------------
def is_table(image_path):
    img = cv2.imread(image_path, 0)
    _, bw = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

    horizontal = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    )
    vertical = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    )

    return (cv2.countNonZero(horizontal) +
            cv2.countNonZero(vertical)) > TABLE_LINE_THRESHOLD


# -----------------------------
# PARAGRAPH OCR
# -----------------------------
def ocr_paragraph(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(
        img,
        config="--psm 3 -c preserve_interword_spaces=1"
    )


# -----------------------------
# TABLE OCR HELPERS
# -----------------------------
def extract_cells(gray):
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    horizontal = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    )
    vertical = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    )

    grid = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(
        grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > MIN_CELL_WIDTH and h > MIN_CELL_HEIGHT:
            cells.append((x, y, w, h))

    return sorted(cells, key=lambda b: (b[1], b[0]))


def ocr_table(image_path, columns=TABLE_COLUMNS):
    gray = cv2.imread(image_path, 0)
    cells = extract_cells(gray)

    table = []
    row = []

    for i, (x, y, w, h) in enumerate(cells):
        cell_img = gray[y:y+h, x:x+w]

        text = pytesseract.image_to_string(
            cell_img,
            config="--psm 6 -c preserve_interword_spaces=1"
        ).strip()

        row.append(text)

        if (i + 1) % columns == 0:
            table.append(row)
            row = []

    return table


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_image(image_path):
    if is_table(image_path):
        return {
            "type": "table",
            "data": ocr_table(image_path)
        }
    else:
        return {
            "type": "paragraph",
            "data": ocr_paragraph(image_path)
        }


# -----------------------------
# SIMPLE QUESTION ANSWERING
# -----------------------------
def ask_table(table, question):
    q = question.lower()
    for row in table:
        if "discharge" in q and "size" in q:
            return row[-1] or "Empty"
    return "Not found"


def ask_paragraph(text, keyword):
    for line in text.splitlines():
        if keyword.lower() in line.lower():
            return line
    return "Not found"


def print_table(table):
    if not table:
        print("Empty table")
        return

    col_count = max(len(r) for r in table)

    # Normalize rows
    norm = []
    for r in table:
        row = [(c if c else "[EMPTY]") for c in r]
        row += [""] * (col_count - len(row))
        norm.append(row)

    # Column widths
    widths = [
        max(len(row[i]) for row in norm)
        for i in range(col_count)
    ]

    def line():
        print("+" + "+".join("-" * (w + 2) for w in widths) + "+")

    line()
    for r in norm:
        print("| " + " | ".join(
            r[i].ljust(widths[i]) for i in range(col_count)
        ) + " |")
        line()


def print_paragraph(text):
    print("\n".join(
        line.rstrip() for line in text.splitlines() if line.strip()
    ))

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    IMAGE = "pdf_pages/page_1.png"
    result = process_image(IMAGE)

    if result["type"] == "table":
        print_table(result["data"])
    else:
        print_paragraph(result["data"])

