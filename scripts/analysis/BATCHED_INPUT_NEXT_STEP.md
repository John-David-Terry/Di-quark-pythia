# Next step: batched binary input (not implemented)

`split_kinematics_extract` still opens one CSV per event. Profiling shows **job list discovery** (`iterdir` over ~1M filenames per side) can dominate **wall time** on large splits, and **per-event `read_csv`** still dominates the worker loop.

## Proposal

1. **Offline (one-time) conversion** of `split_90_10/altered` and `split_90_10/unchanged` event CSVs into:
   - **Parquet** (or Arrow IPC) with **the same logical columns** as today’s event files, keyed by `event_id` (and `particle_index` within event), **or**
   - **Chunked Parquet** files each holding many events (e.g. row group per event or `event_id` column for filter pushdown).

2. **Reader path** in the kinematics script:
   - Build the job list from **manifest** / chunk index (small JSON next to Parquet) so benchmark mode does not scan million-entry directories.
   - Load one event with **`pd.read_parquet(..., columns=USECOLS, filters=[('event_id','==',id)])`** or slice from a preloaded table — **no physics changes**, same column semantics.

3. **Artifacts**: preserve existing merged CSV columns; workers may still write CSV shards or switch to Parquet for merge efficiency.

## Dependencies

- `pyarrow` (or `fastparquet`) for Parquet; keep optional import with clear error if missing.

## Out of scope here

- Changing PYTHIA or Breit logic; only I/O and job discovery.
