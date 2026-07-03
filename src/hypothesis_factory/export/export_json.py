from __future__ import annotations

import csv
import json
from io import StringIO

from hypothesis_factory.models import Hypothesis


def hypotheses_to_json(hypotheses: list[Hypothesis]) -> str:
    return json.dumps([item.to_dict() for item in hypotheses], ensure_ascii=False, indent=2)


def hypotheses_to_csv(hypotheses: list[Hypothesis]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["id", "title", "target_kpi", "zone_id", "zone_type", "final_score"])
    writer.writeheader()
    for item in hypotheses:
        zone = item.origin_uncertainty_zone
        writer.writerow(
            {
                "id": item.id,
                "title": item.title,
                "target_kpi": item.target_kpi,
                "zone_id": zone.get("id"),
                "zone_type": zone.get("type"),
                "final_score": item.scores.get("final_score"),
            }
        )
    return buffer.getvalue()

