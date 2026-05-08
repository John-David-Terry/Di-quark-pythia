# Moderate-statistics validation report: branch classification vs Breit hemisphere

## High-level summary

### Runs performed

1. **ISRFSR_ON, 1000 accepted events** — full output captured.
2. **ISRFSR_OFF, 1000 accepted events** — full output captured.
3. **ISRFSR_ON, 5000 accepted events** — key cross-tabulation and fractions captured.

### Main result: remnant branch empty in this sample

In all runs, **no event had a plausible remnant branch** under the current definition (“proton daughters not on the struck branch”). So:

- **Remnant/target consistency cannot be tested** with these runs: there are zero remnant-branch-reachable hadrons.
- **Struck/current consistency can be tested:** struck-branch-reachable hadrons are predominantly in the Breit current hemisphere (pz_breit < 0), with fractions ~75–78% depending on label and sample.

### Struck/current validation

- **ISRFSR_ON (1k):** f_struck,current = 3007/3965 = **0.758** (all hadrons and π⁻ only; only struck hadrons appear).
- **ISRFSR_OFF (1k):** f_struck,current = 2325/2991 = **0.777**.
- **ISRFSR_ON (5k):** f_struck,current = 15040/20134 = **0.747**.

So the **struck branch clearly prefers the Breit current hemisphere** (pz_breit < 0); the majority is consistently in the 75–78% range.

### Remnant/target validation

- **Not testable in this validation:** N(remnant_branch_reachable) = 0 for every run. The current remnant-branch definition (proton daughters not on the struck branch) yields no remnant seeds in these PYTHIA samples. So f_remnant,target is undefined (0/0).

### Breit transform and other checks

- **Breit transform:** No failures. Events with failed Breit transform = 0 for all runs; fraction failed = 0.
- **LAB vs Breit (π⁻):** Forward π⁻ (pz_lab > 0) largely lie in the Breit target hemisphere: P(pz_breit > 0 | forward π⁻) ≈ 0.90–0.91 in the 1k runs.

### Conclusion

- **Struck/current:** The ancestry-based struck-branch classification is consistent with the Breit current hemisphere (pz_breit < 0); ~75–78% of struck-branch hadrons are in the current hemisphere.
- **Remnant/target:** The remnant-side validation could not be run because the current script found no remnant branch in any of the 1000 or 5000 event samples. Improving or auditing the remnant-branch definition (e.g. as in the next-step validation plan) is needed before remnant/target consistency can be measured.

---

## Low-level summary for ChatGPT

### Commands run

```bash
# 1. ISRFSR_ON, 1000 events
python3.11 scripts/generation/test_hadron_progenitor_tracing.py --labels ISRFSR_ON --n-events 1000 --max-debug-events 0

# 2. ISRFSR_OFF, 1000 events
python3.11 scripts/generation/test_hadron_progenitor_tracing.py --labels ISRFSR_OFF --n-events 1000 --max-debug-events 0

# 3. ISRFSR_ON, 5000 events
python3.11 scripts/generation/test_hadron_progenitor_tracing.py --labels ISRFSR_ON --n-events 5000 --max-debug-events 0
```

All from project root: `Di-quark-pythia`.

### Labels and accepted-event counts

| Label       | Requested | Accepted events |
|------------|-----------|------------------|
| ISRFSR_ON  | 1000      | 1000             |
| ISRFSR_OFF | 1000      | 1000             |
| ISRFSR_ON  | 5000      | 5000             |

### Breit transform failures

- **ISRFSR_ON 1k:** 0 / 1000 (0%).
- **ISRFSR_OFF 1k:** 0 / 1000 (0%).
- **ISRFSR_ON 5k:** 0 / 5000 (0%).

No warning signs; Breit transform was always valid when used.

### Compact summary by run

---

#### Run 1: ISRFSR_ON, 1000 accepted events

- **Accepted events:** 1000  
- **Total traced hadrons:** 3965  
- **Total traced π⁻:** 3965  
- **Total traced forward π⁻ (pz_lab > 0):** 578  
- **Events with plausible remnant branch:** 0  
- **Breit transform failed:** 0 / 1000 (0%)

**All hadrons** (all are struck-branch-reachable; no remnant or both):

- **Struck/current:** 3007 / 3965 = **0.758**  
- **Remnant/target:** 0 / 0 (N/A)

**π⁻ only:**

- **Struck/current:** 3007 / 3965 = **0.758**  
- **Remnant/target:** 0 / 0 (N/A)

**Forward π⁻ only:**

- Forward π⁻ with pz_breit > 0: 528 / 578 ⇒ forward π⁻ in current (pz_breit < 0): 50.
- **Struck/current (forward π⁻):** 50 / 578 = **0.087** (i.e. 91.3% of forward π⁻ are in target; they are still classified as struck-branch-reachable).
- **Remnant/target:** 0 / 0 (N/A)

**Both-branch:** 0 hadrons.

---

#### Run 2: ISRFSR_OFF, 1000 accepted events

- **Accepted events:** 1000  
- **Total traced hadrons:** 2991  
- **Total traced π⁻:** 2991  
- **Total traced forward π⁻:** 466 (from Forward π⁻ with pz_breit > 0: 420/466)  
- **Events with plausible remnant branch:** 0  
- **Breit transform failed:** 0 / 1000 (0%)

**All hadrons:**

- **Struck/current:** 2325 / 2991 = **0.777**  
- **Remnant/target:** 0 / 0 (N/A)

**π⁻ only:**

- **Struck/current:** 2325 / 2991 = **0.777**  
- **Remnant/target:** 0 / 0 (N/A)

**Forward π⁻ only:**

- **Struck/current (forward π⁻):** (466−420) / 466 = 46/466 = **0.099**  
- **Remnant/target:** 0 / 0 (N/A)

**Both-branch:** 0 hadrons.

---

#### Run 3: ISRFSR_ON, 5000 accepted events

- **Accepted events:** 5000  
- **Total traced hadrons:** 20134  
- **Total traced π⁻:** 20134  
- **Total traced forward π⁻:** not extracted; script output truncated.  
- **Events with plausible remnant branch:** 0  
- **Breit transform failed:** 0 / 5000 (0%)

**All hadrons:**

- **Struck/current:** 15040 / 20134 = **0.747**  
- **Remnant/target:** 0 / 0 (N/A)

**π⁻ only:**

- **Struck/current:** 15040 / 20134 = **0.747**  
- **Remnant/target:** 0 / 0 (N/A)

**Forward π⁻ only:** raw counts not in captured output; fraction struck in current for all π⁻ is 0.747.

**Both-branch:** 0 hadrons.

---

### Numerator/denominator summary

| Run            | Category    | f_struck,current (num/den) | f_remnant,target (num/den) |
|----------------|------------|-----------------------------|-----------------------------|
| ISRFSR_ON 1k  | all hadrons| 3007 / 3965 = 0.758         | 0 / 0 (N/A)                 |
| ISRFSR_ON 1k  | π⁻ only    | 3007 / 3965 = 0.758         | 0 / 0 (N/A)                 |
| ISRFSR_ON 1k  | forward π⁻ | 50 / 578 = 0.087             | 0 / 0 (N/A)                 |
| ISRFSR_OFF 1k | all hadrons| 2325 / 2991 = 0.777         | 0 / 0 (N/A)                 |
| ISRFSR_OFF 1k | π⁻ only    | 2325 / 2991 = 0.777         | 0 / 0 (N/A)                 |
| ISRFSR_OFF 1k | forward π⁻ | 46 / 466 = 0.099             | 0 / 0 (N/A)                 |
| ISRFSR_ON 5k  | all hadrons| 15040 / 20134 = 0.747       | 0 / 0 (N/A)                 |
| ISRFSR_ON 5k  | π⁻ only    | 15040 / 20134 = 0.747       | 0 / 0 (N/A)                 |

### Low-statistics caveats

- **Remnant:** No remnant-branch hadrons in any run, so remnant/target fractions are undefined. The remnant-branch definition (proton daughters not on struck branch) yields zero remnant seeds in these samples; this is a definition/event-structure issue, not a statistics issue.
- **Struck:** Statistics are sufficient. Thousands of struck-branch hadrons and π⁻; fractions are stable between 1k and 5k (ISRFSR_ON) and between ISRFSR_ON and ISRFSR_OFF.
- **Forward π⁻:** In the 1k runs, forward π⁻ number in the hundreds (466–578); struck/current for forward π⁻ is a minority (≈9–10%) because most forward π⁻ have pz_breit > 0 (target hemisphere), while they are still classified as struck-branch-reachable.

### Interpretation

- **Struck branch → Breit current hemisphere:** Validated. A clear majority (≈75–78%) of struck-branch-reachable hadrons have pz_breit < 0. The ancestry-based struck classification is consistent with the Breit current (fragmentation) side.
- **Remnant branch → Breit target hemisphere:** Not testable. No remnant-branch hadrons were found in 1000 or 5000 event samples with the current definition. The next-step validation (remnant-seed audit and alternative definitions) is needed to enable this check.
