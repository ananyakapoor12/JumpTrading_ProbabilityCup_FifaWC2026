"""
REST client for SportsPredict Probability Cup.
Base URL: https://api.sportspredict.com/api/v1
Auth:      Bearer sp_live_<key>

Discovery flow (required before predictions):
  event  →  lobby (auto-join)  →  match  →  markets  →  submit batch
"""

import requests


BASE_URL = "https://api.sportspredict.com/api/v1"


# FIFA 3-letter codes that differ from the first 3 letters of the English name.
_FIFA_CODES = {
    "Netherlands": "NED", "Morocco": "MAR", "Ivory Coast": "CIV",
    "South Korea": "KOR", "DR Congo": "COD", "Saudi Arabia": "KSA",
    "Czech Republic": "CZE", "Czechia": "CZE", "Cape Verde": "CPV",
    "Bosnia": "BIH", "Bosnia and Herzegovina": "BIH",
    "Algeria": "ALG", "Austria": "AUT", "Croatia": "CRO",
    "Switzerland": "SUI", "Australia": "AUS", "Senegal": "SEN",
    "Ghana": "GHA", "Colombia": "COL", "Ecuador": "ECU",
}


class KapbotClient:
    def __init__(self, api_key: str):
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._event_id = None
        self._lobby_id = None

    # ── Low-level HTTP ────────────────────────────────────────────────

    def _get(self, path, **params):
        r = self._session.get(BASE_URL + path,
                              params={k: v for k, v in params.items() if v is not None},
                              timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path, body=None):
        r = self._session.post(BASE_URL + path, json=body, timeout=15)
        r.raise_for_status()
        return r.json()

    def _patch(self, path, body=None):
        r = self._session.patch(BASE_URL + path, json=body, timeout=15)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _as_list(data, key):
        """Normalise API response to a list regardless of envelope shape."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get(key, [])
        return []

    # ── Event / lobby discovery (cached) ─────────────────────────────

    def event_id(self) -> str | None:
        """Active Probability Cup event ID. Fetched once, then cached."""
        if self._event_id:
            return self._event_id
        try:
            data = self._get("/events", limit=10)
            for ev in self._as_list(data, "events"):
                if (ev.get("type") == "probability" and
                        ev.get("status") in ("active", "upcoming", "open")):
                    self._event_id = ev["id"]
                    return self._event_id
            # Fallback: first event in list
            evs = self._as_list(data, "events")
            if evs:
                self._event_id = evs[0]["id"]
        except Exception as e:
            print(f"  [api] event lookup failed: {e}")
        return self._event_id

    def lobby_id(self) -> str | None:
        """
        Joined lobby ID. Auto-joins the first available lobby if not already
        a member. Cached after first successful call.
        """
        if self._lobby_id:
            return self._lobby_id
        eid = self.event_id()
        if not eid:
            return None
        try:
            data = self._get("/lobbies", event_id=eid)
            for lobby in self._as_list(data, "lobbies"):
                lid = lobby["id"]
                if not lobby.get("joined"):
                    try:
                        self._post(f"/lobbies/{lid}/join")
                    except requests.HTTPError as e:
                        if e.response is not None and e.response.status_code == 409:
                            pass  # 409 = already joined — that's fine
                        else:
                            raise
                self._lobby_id = lid
                return self._lobby_id
        except Exception as e:
            print(f"  [api] lobby lookup failed: {e}")
        return self._lobby_id

    # ── Matches / markets ─────────────────────────────────────────────

    def get_matches(self) -> list:
        """List all matches with open markets for the active event + lobby."""
        eid = self.event_id()
        lid = self.lobby_id()
        if not eid or not lid:
            return []
        try:
            data = self._get("/matches", event_id=eid, lobby_id=lid)
            return self._as_list(data, "matches")
        except Exception as e:
            print(f"  [api] matches fetch failed: {e}")
            return []

    def find_match(self, home_team: str) -> dict | None:
        """
        Find a match dict by home team name.
        Tries: exact name → FIFA 3-letter code → first-3-letter prefix.
        Returns None if no open match is found.
        """
        code = _FIFA_CODES.get(home_team, home_team[:3]).upper()
        for m in self.get_matches():
            name = m.get("name", "")
            if home_team in name or code in name.upper():
                return m
        return None

    def get_markets(self, match_id: str) -> list:
        """List all binary markets for a given match in the active lobby."""
        lid = self.lobby_id()
        if not lid:
            return []
        try:
            data = self._get("/markets", lobby_id=lid, match_id=match_id)
            return self._as_list(data, "markets")
        except Exception as e:
            print(f"  [api] markets fetch failed: {e}")
            return []

    # ── Predictions ───────────────────────────────────────────────────

    def submit_batch(self, predictions: list) -> dict | None:
        """
        Batch-submit up to 50 predictions.

        Each entry must be:
            {"market_id": str, "lobby_id": str, "probability": int}  # 1–99

        Returns the API response dict (keys: total, succeeded, failed, results).
        """
        if not predictions:
            return None
        try:
            return self._post("/predictions/batch", {"predictions": predictions})
        except Exception as e:
            print(f"  [api] batch submit failed: {e}")
            return None

    def update_prediction(self, prediction_id: str, probability: int) -> dict | None:
        """Revise a single prediction before market close."""
        try:
            return self._patch(f"/predictions/{prediction_id}",
                               {"probability": probability})
        except Exception as e:
            print(f"  [api] update prediction failed: {e}")
            return None

    def get_predictions(self) -> list:
        """List your submitted predictions for the active lobby."""
        lid = self.lobby_id()
        if not lid:
            return []
        try:
            data = self._get("/predictions", lobby_id=lid)
            return self._as_list(data, "predictions")
        except Exception as e:
            print(f"  [api] predictions fetch failed: {e}")
            return []

    def submit_or_update_batch(self, predictions: list) -> dict:
        """
        Smart batch submit: for each prediction, checks whether you
        already have one on that market. If so, calls update_prediction
        (PATCH) instead of submit (POST), so re-running the pipeline
        right up until kickoff always refreshes your numbers instead
        of failing with "Prediction for this market already exists".

        Each entry in `predictions` must be:
            {"market_id": str, "lobby_id": str, "probability": int}

        Returns: {"succeeded": int, "failed": int, "updated": int,
                  "created": int, "results": [...]}
        """
        existing = self.get_predictions()
        existing_by_market = {p["market_id"]: p for p in existing}

        to_create = []
        to_update = []  # (prediction_id, market_id, probability)

        for p in predictions:
            mid = p["market_id"]
            if mid in existing_by_market:
                pred = existing_by_market[mid]
                # Skip if already settled — can't update a closed market
                if pred.get("market_status") == "settled":
                    continue
                to_update.append((pred["id"], mid, p["probability"]))
            else:
                to_create.append(p)

        results = []
        updated, created, failed = 0, 0, 0

        # Updates — one PATCH call per market (no batch update endpoint)
        for pred_id, mid, prob in to_update:
            r = self.update_prediction(pred_id, prob)
            if r:
                updated += 1
                results.append({"market_id": mid, "action": "updated",
                                "success": True, "probability": prob})
            else:
                failed += 1
                results.append({"market_id": mid, "action": "updated",
                                "success": False, "error": "PATCH failed"})

        # New submissions — one batch POST for everything not yet predicted
        if to_create:
            r = self.submit_batch(to_create)
            if r:
                created += r.get("succeeded", len(to_create))
                failed  += r.get("failed", 0)
                results.extend(r.get("results", []))
            else:
                failed += len(to_create)
                results.append({"action": "created", "success": False,
                                "error": "batch submit failed"})

        return {
            "succeeded": updated + created,
            "failed":    failed,
            "updated":   updated,
            "created":   created,
            "results":   results,
        }

    def get_results(self) -> list:
        """List settled predictions with Brier scores for the active lobby."""
        lid = self.lobby_id()
        if not lid:
            return []
        try:
            data = self._get("/results", lobby_id=lid)
            return self._as_list(data, "results")
        except Exception as e:
            print(f"  [api] results fetch failed: {e}")
            return []