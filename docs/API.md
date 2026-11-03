# API reference

Base URL: `http://localhost:5000`. Requests use JSON. The server runs through `python src-backend/run.py` with the existing single-thread Waitress configuration.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Process liveness (`{"status":"ok"}`), not model readiness |
| POST | `/recommend` | Profile ranking or popularity fallback |
| GET | `/similar_items?item_id=1&k=6` | Neighbor films; `k` from 1 to 100 |
| POST | `/event` | Record interaction and update a session vector |
| GET | `/lantern/profile?user_id=demo-123` | Genre summary, recent interactions, vector norm |
| POST | `/lantern/recommend` | Taste ranking or popularity fallback |
| POST | `/lantern/reset` | Clear the given session's in-memory taste |

## Profile ranking

```json
{"gender":"F","age":25,"occupation":12}
```

Age codes: `1`, `18`, `25`, `35`, `45`, `50`, `56`; `-1` means unknown. Occupation: `0..20` or `-1`. Gender: `F`, `M`, `UNKNOWN`. An unknown field selects popularity fallback. Invalid profile codes return HTTP 400.

```json
{"recommendations":[[2858,"American Beauty (1999)",0.683]],"source":"model"}
```

The example score is illustrative; do not rely on exact ranking/float values across runtime hardware. `source` is `model` or `popular`. Items in this route are `[item_id, title, score]` arrays.

## Events and session taste

```json
{"user_id":"demo-123","item_id":1,"event":"like"}
```

Supported events: `click`, `like`, `finish`, `dismiss`. Unknown item IDs and event names return HTTP 400. The UI exposes likes. Repeated API events are not deduplicated; they update the vector again. The current page disables a film's like button after a successful like.

The event response contains `ok` and a `recommendations` array of objects with `item_id`, `title`, `score`. Taste recommendation requests use:

```json
{"user_id":"demo-123","exclude":[1,2]}
```

`/lantern/recommend` includes `note: "popular_fallback"` when the session is missing or its vector norm is below 0.2. Fallback items are arrays; vector-based items are objects. The frontend normalizes both in `src/lib/api.js`. Similarity responses also use objects under `similar`.

Reset requests contain `{"user_id":"demo-123"}`. User IDs are arbitrary demo identifiers, not authenticated identities. State is shared only inside one running Python process and lost on restart. This API is intended for local demonstration, not untrusted multi-user hosting.

The prototype generally returns `{"error":"..."}` with HTTP 400 for route failures, including some server/runtime errors. A production API would separate client validation errors from server errors and use a consistent item schema.
