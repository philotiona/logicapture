from PIL import Image, ImageDraw, ImageFont
import numpy as np

BLUE = "#002FA7"


def group_board(images, captions, columns=3, tile=150):
    """One ordered contact sheet instead of nested columns and repeated panels."""
    rows = (len(images)+columns-1)//columns
    gutter, label_height = 12, 25
    board = Image.new("RGB", (columns*tile+(columns-1)*gutter, rows*(tile+label_height)+(rows-1)*gutter), "white")
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=12)
    for i, (image, caption) in enumerate(zip(images, captions)):
        image = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")
        image.thumbnail((tile-12, tile-12), Image.Resampling.LANCZOS)
        x, y = (i%columns)*(tile+gutter), (i//columns)*(tile+label_height+gutter)
        draw.text((x, y+3), caption, fill="black", font=font)
        draw.rectangle((x, y+label_height, x+tile-1, y+label_height+tile-1), outline="#BBBBBB")
        board.paste(image, (x+(tile-image.width)//2, y+label_height+(tile-image.height)//2))
    return board


def panel_summary(rule, scene):
    """Human-readable evaluation evidence; full recursive AST remains available."""
    evidence = rule.explain(scene)
    if rule.children:
        return [{"Condition": item["rule"], "Result": "Unresolved" if item["matches"] is None else "True" if item["matches"] else "False"}
                for item in evidence.get("steps", [])]
    rows = [{"Measurement": "Objects matching the selector", "Value": str(evidence.get("count", 0))},
            {"Measurement": "Matching object IDs", "Value": ", ".join(map(str, evidence.get("selected_ids", []))) or "None"}]
    if "witnesses" in evidence:
        rows.append({"Measurement": "Relation pairs", "Value": ", ".join(f"{a} → {b}" for a, b in evidence["witnesses"]) or "None"})
    rows.append({"Measurement": "Rule evaluation", "Value": "Unresolved" if evidence["matches"] is None else "True → A" if evidence["matches"] else "False → B"})
    return rows
