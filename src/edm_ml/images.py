from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from docx import Document


def inventory_and_extract(docx_path: str | Path, output_dir: str | Path) -> list[dict]:
    docx_path, output_dir = Path(docx_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = Document(docx_path)
    labels = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    records = []
    with zipfile.ZipFile(docx_path) as archive:
        media = sorted(n for n in archive.namelist() if n.startswith("word/media/"))
        for i, member in enumerate(media, start=1):
            destination = output_dir / Path(member).name
            destination.write_bytes(archive.read(member))
            records.append({"index": i, "file": destination.name,
                            "bytes": destination.stat().st_size,
                            "document_labels": " | ".join(labels)})
    with (output_dir / "image_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys() if records else ["index", "file"])
        writer.writeheader(); writer.writerows(records)
    return records

