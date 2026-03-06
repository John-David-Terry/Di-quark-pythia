# Eta Analysis Config Checklist

## Original (generate_pdf_plots_new.py run_eta_analysis_and_plot)

| Setting | Value |
|---------|-------|
| Beams:idA | 2212 (proton) |
| Beams:idB | 11 (electron) |
| Beams:eA | 275.0 |
| Beams:eB | 18.0 |
| Beams:frameType | 2 |
| PartonLevel:ISR | on |
| PartonLevel:FSR | on |
| HadronLevel:all | on |
| **ColourReconnection:reconnect** | **on** |
| PhaseSpace:Q2Min | 16.0 |
| WeakBosonExchange:ff2ff(t:gmZ) | on |
| HardQCD:all | off |
| PDF:lepton | off |

## Cached Shards (pythia_finalstate_raw/ISRFSR_ON)

From meta.json: E_e=18, E_p=275, Q2_min=16, isr_fsr_on=true, seed=12346.
**meta.json does NOT store ColourReconnection or Beams:idA/idB.**

From conversation history, generate_events_raw.py used:
| Setting | Value |
|---------|-------|
| Beams:idA | 11 (electron) ← **OPPOSITE** |
| Beams:idB | 2212 (proton) |
| ColourReconnection:reconnect | **off** ← **MISMATCH** |

## Root Cause of Plot Mismatch

1. **ColourReconnection**: Original eta uses CR **ON**. Cached shards were generated with CR **OFF**.
   - CR changes hadronization and thus eta distribution significantly.

2. **Beam order**: Shards use idA=11, idB=2212 (e +z, p -z). Original uses idA=2212, idB=11 (p +z, e -z).
   - analyze_events_raw applies FLIP_Z to correct frame, but the underlying events are still from different physics (CR off).

## Fix

**Option A**: Regenerate eta dataset with EXACT same config as original:
- Beams:idA=2212, idB=11
- ColourReconnection:reconnect=on
- Create label ETA_ON_CRON and generate that dataset.

**Option B**: Run original with CR off to compare apples-to-apples (for debugging only).

## Comparator Result (debug_compare_cached_vs_pythia_eta.py)

When running on the **same** PYTHIA events (original config, packed in-memory):
- **eta matches exactly** (chosen_pid, eta identical)
- Small differences in x, Q, LT, q_tr are from **float32 vs float64** precision (~1e-5)
- Analysis logic is correct; the mismatch is **generator config**, not analysis code.
