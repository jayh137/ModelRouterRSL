# First release validation

Validated on 2026-10-17 using Python 3.13.12 on Windows.

- `python -m unittest discover -s tests -v`: **50 tests passed**.
- `python -m app.rsl.worker demo`: collection, history training, actual recommender replay, activation and rollback completed successfully in a temporary database.
- `python -m compileall -q app tests`: passed.
- `python -m pip check`: no broken requirements.
- `git diff --check`: passed.
- Minimal application startup with empty providers: `/health` returned 200 with RSL both disabled and enabled.
- Admin authorization: missing credentials, incorrect credentials and the internal service key were rejected. The admin key returned 404 while RSL was disabled and 200 while enabled.

The test suite covers feedback validation and idempotency, per-profile version isolation, unknown costs, streaming interruption, retries, cascade and judge telemetry, training lineage, held-out data reservation, configuration drift, failed/stale evaluation rejection, and rollback to the original baseline.

The demo uses synthetic observations and no external provider calls. It validates software behavior, **not real model quality, savings, or production performance**. Production promotion requires independently measured held-out cases. See [RSL operations](RSL.md) for scope and known limitations.

GitHub Actions additionally runs the suite and demo on Python 3.11, 3.12 and 3.13. Check the repository Actions page for those results; the local result above covers Python 3.13 only.
