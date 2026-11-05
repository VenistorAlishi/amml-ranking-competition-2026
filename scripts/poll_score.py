"""Poll latest Kaggle submission score without printing tokens."""
from __future__ import annotations

import time

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    comp = "amml-ranking-competition-2026"
    for attempt in range(24):
        subs = api.competition_submissions(comp)
        # newest first typically
        rows = []
        for s in subs[:8]:
            rows.append(
                {
                    "ref": getattr(s, "ref", None),
                    "date": str(getattr(s, "date", "")),
                    "desc": getattr(s, "description", None) or getattr(s, "fileName", ""),
                    "status": getattr(s, "status", None),
                    "publicScore": getattr(s, "publicScore", None),
                }
            )
        print(f"attempt={attempt}", flush=True)
        for r in rows:
            print(r, flush=True)
        top = rows[0] if rows else None
        if top and top["publicScore"] not in (None, "", "None") and str(top["status"]).lower() in (
            "complete",
            "finished",
        ):
            print("LATEST_PUBLIC", top["publicScore"], flush=True)
            return
        if top and str(top.get("status", "")).lower() in ("error", "failed"):
            print("LATEST_FAILED", top, flush=True)
            return
        time.sleep(15)
    print("TIMEOUT waiting for score", flush=True)


if __name__ == "__main__":
    main()
