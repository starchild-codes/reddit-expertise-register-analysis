"""Create non-text public audit outputs from a private historical audit CSV.

The source file is not distributed. This script strips titles and other text
before writing the release-ready AI-relevance decision file.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "current" / "ai_relevance_decisions.csv"

def main(source: Path) -> None:
    with source.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"subreddit", "expertise_tier", "ai_relevant_screen"}
    if not required.issubset(rows[0] if rows else {}):
        raise ValueError(f"Source must include {sorted(required)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        fields = ["post_row_id", "subreddit", "community_orientation", "ai_relevance_decision", "decision_method", "decision_note"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            retained = str(row["ai_relevant_screen"]).strip().lower() == "true"
            writer.writerow({
                "post_row_id": i,
                "subreddit": row["subreddit"],
                "community_orientation": row["expertise_tier"],
                "ai_relevance_decision": "retain" if retained else "exclude_non_ai",
                "decision_method": "preserved_conservative_content_screen",
                "decision_note": "Preserved audit decision; text is intentionally not released.",
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Private historical audit CSV; never commit it.")
    main(parser.parse_args().source)
