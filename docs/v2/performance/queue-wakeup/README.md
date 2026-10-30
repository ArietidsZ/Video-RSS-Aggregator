# Local queue wakeup experiment (2026-10-30)

Baseline: `f938dd09510ecbd16fd62ca8428adb2401785b69`. Candidate: the sibling commit containing this report.

Scope: real temporary SQLite, fake no-op processing, no network/model/device calls. Run `PROBE_LABEL=<label> PYTHONPATH=<checkout> python docs/v2/performance/queue-wakeup/probe.py` with the project's locked Python dependencies. Each worker-count trial contains 20 sparse jobs, a 32-job burst and 10 enqueues of the same future job. Sources and raw observations are included. The baseline is an unmodified worktree, not a simulated polling implementation.

| Workers | Baseline sparse median/p95 ms | Candidate sparse median/p95 ms | Baseline/candidate total empty claims |
| --- | --- | --- | --- |
| 2 (default) | 973.58 / 977.20 | 1.41 / 4.37 | 44 / 74 |
| 8 | 795.93 / 1754.34 | 2.98 / 5.95 | 120 / 392 |
| 32 | 973.07 / 974.51 | 2.51 / 5.22 | 704 / 1664 |

All trials processed exactly 52 distinct jobs and left zero worker tasks pending after shutdown. p95 is nearest-rank over only 20 sparse observations; this is not production p95. Host contention was uncontrolled (including concurrent unrelated validation), so retain the outliers, do not infer their cause, and do not compare total CPU across different wall durations. This experiment establishes local notification responsiveness, not model throughput, inference latency, sustained memory, idle CPU or end-to-end download speed.

## Design and tradeoff

Workers capture the current notification epoch before the durable claim. Enqueue rotates and signals that epoch only after commit and only for a stored queued/due job. Cancellation cannot clear another worker's signal. SQLite still decides ownership; notifications never carry jobs. The one-second fallback remains for other queue instances/processes and future retry deadlines. Schema, contracts, retry limits and processing concurrency are unchanged.

Broadcast schedules every idle worker of the matching kind. Burst traffic can therefore cause extra losing claims, especially at high worker counts; the table exposes this cost rather than claiming a universal CPU win. The intermediate broadcast version generated 20/80/320 useless empty claims for the ten future-job enqueues at W=2/8/32. Matching the stored deadline to claim's lexical ordering reduced that isolated delta to zero in all three final trials. Existing legacy timestamp-format limitations are not repaired by this change.

A more complex FIFO notifier was not adopted: cancellation handoff and another consumer winning a claim would require additional correctness machinery. Measure real workload pressure before making that tradeoff. Current tests cover claim/wait races, dedup, transaction rollback, future deadlines, stale recovery, both runtime worker kinds, cancellation isolation and epoch collection. A gate-based mutation test failed for both worker kinds when notification waiting was replaced by the old sleep; its success does not depend on a subsecond performance threshold.
