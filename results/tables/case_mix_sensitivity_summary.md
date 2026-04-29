# Case-mix sensitivity of the direction asymmetry

## Question

Does the 2.6-fold difference between mean MOVER-trained degradation (13.9%) and mean INSPIRE-trained degradation (5.4%) persist when case-mix is matched across the two external test sets?

## Three matched framings

| Framing | Description | Matched asymmetry (pp) | 95% CI | Bootstrap p | % of original | Interpretation |
|---|---|---:|---|---:|---:|---|
| framing_match_inspire_casemix | Matched to INSPIRE case-mix (90.4/9.6) | +17.31 | +12.52 to +21.60 | 0.0010 | 203% | case_mix_is_not_primary_driver |
| framing_match_mover_casemix | Matched to MOVER case-mix (36.5/63.5) | +11.56 | +9.69 to +13.43 | 0.0010 | 136% | case_mix_is_not_primary_driver |
| framing_balanced_50_50 | Balanced 50/50 case-mix | +10.90 | +9.03 to +12.90 | 0.0010 | 128% | case_mix_is_not_primary_driver |

**Baseline (unmatched, for reference):** +8.53pp (95% CI: +6.91 to +10.24pp; bootstrap p=0.0010).

## Headline interpretation

**Case-mix does not drive the asymmetry — in fact, matching case-mix *increases* it.** In every matched framing the asymmetry grows to 128–203% of the unmatched baseline while remaining highly statistically significant (bootstrap p=0.001 in all three framings). This pattern is consistent with training-population diversity being the driver: when each group of models is evaluated on an external cohort whose case-mix more closely resembles its own training distribution, the MOVER-trained degradation either rises or holds while the INSPIRE-trained degradation drops (or goes negative), amplifying the gap.

## Caveats

* Subsampling retains at most ~23,000 MOVER and ~19,000 INSPIRE cases in Framing A; Framing B retains ~42,000 and ~25,000 respectively. Power is lower than the full-cohort baseline (127,413 and 57,545) but remains sufficient to detect the 8.5pp baseline effect.
* Internal AUCs are treated as fixed (Phase-2 OOF point estimates). Uncertainty in internal AUCs is not propagated.
* Subsampling is deterministic (seed=42); bootstrap is within-subsample. A double bootstrap (vary subsample + case-level) would give a slightly wider CI; not performed here.
