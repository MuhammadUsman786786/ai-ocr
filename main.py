import os
import time

import easyocr

from pdf2image import convert_from_path

reader = easyocr.Reader(['en'], gpu=True)


def export_pdf_pages():
    pdf_path = "scanned.pdf"
    output_folder = "pdf_pages"
    dpi = 300

    os.makedirs(output_folder, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi)
    for i, page in enumerate(pages, start=1):
        image_path = os.path.join(output_folder, f"page_{i}.png")
        page.save(image_path, "PNG")
        print(f"Saved: {image_path}")

    print("All pages exported successfully!")


def ocr_image():
    start_time = time.time()
    allowlist_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-/:._()&'
    result = reader.readtext("pdf_pages/page_1.png", allowlist=allowlist_chars)
    texts = [text_info[1] for text_info in result]
    ocr_text = "\n".join(texts)
    print(ocr_text)
    end_time = time.time()
    print(f"\nOCR runtime: {end_time - start_time:.2f} seconds")


def process_document_with_tables(image_path):
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(image_path)

    # --- Step 1: Extract text and bounding boxes ---
    boxes = []
    for bbox, text, conf in result:
        x_min = min(pt[0] for pt in bbox)
        y_min = min(pt[1] for pt in bbox)
        x_max = max(pt[0] for pt in bbox)
        y_max = max(pt[1] for pt in bbox)
        boxes.append({'text': text, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max})

    # --- Step 2: Sort boxes by top-left y (top to bottom), then x (left to right) ---
    boxes_sorted = sorted(boxes, key=lambda b: (b['y_min'], b['x_min']))

    # --- Step 3: Cluster boxes into rows/blocks ---
    row_thresh = 10  # pixel threshold for same row
    blocks = []
    current_row = []
    current_y = -1

    for box in boxes_sorted:
        if current_y == -1:
            current_y = box['y_min']
        if abs(box['y_min'] - current_y) < row_thresh:
            current_row.append(box)
        else:
            blocks.append(current_row)
            current_row = [box]
            current_y = box['y_min']
    if current_row:
        blocks.append(current_row)

    # --- Step 4: Process each block ---
    processed_blocks = []

    for block in blocks:
        # Detect if block is table-like (more than 1 unique x-coordinate)
        x_positions = sorted(set([b['x_min'] for b in block]))
        if len(x_positions) > 1:
            # Table-like: assign each box to nearest column
            cols = len(x_positions)
            row_cells = [''] * cols
            for b in block:
                col_idx = min(range(cols), key=lambda i: abs(x_positions[i] - b['x_min']))
                row_cells[col_idx] = b['text']
            processed_blocks.append({'type': 'table_row', 'cells': row_cells})
        else:
            # Paragraph: join all text in reading order
            paragraph_text = ' '.join([b['text'] for b in block])
            processed_blocks.append({'type': 'paragraph', 'text': paragraph_text})

    return processed_blocks

# --- Example usage ---
blocks = process_document_with_tables("pdf_pages/page_1.png")
for blk in blocks:
    if blk['type'] == 'table_row':
        print("Table Row:", blk['cells'])
    else:
        print("Paragraph:", blk['text'])
