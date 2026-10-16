# OCSR benchmark: phase1_benchmark

- Finished: 2026-10-15T03:58:10+00:00 | config hash: 0d2eefb7283ec505 | GPU: NVIDIA H200

## Overall (all scorable structures)

| tool | dataset | n | exact match [95% CI] | no-stereo | InChIKey conn. | mean Tanimoto | valid SMILES | img/s |
|---|---|---|---|---|---|---|---|---|
| biominer_molglyph | clef | 454 | 0.780 [0.742, 0.817] | 0.874 | 0.874 | 0.962 | 1.000 | 14.2 |
| biominer_molglyph | uob | 500 | 0.888 [0.860, 0.916] | 0.898 | 0.964 | 0.958 | 1.000 | 27.0 |
| biominer_molglyph | uspto | 489 | 0.730 [0.691, 0.767] | 0.814 | 0.816 | 0.936 | 0.984 | 13.9 |
| biominer_molglyph | jpo | 449 | 0.800 [0.764, 0.833] | 0.829 | 0.829 | 0.914 | 0.955 | 14.3 |
| biominer_molglyph | biovista_ocsr | 1000 | 0.760 [0.735, 0.788] | 0.870 | 0.872 | 0.945 | 0.995 | 17.6 |
| molnextr | clef | 454 | 0.764 [0.725, 0.804] | 0.866 | 0.866 | 0.946 | 0.987 | 1.3 |
| molnextr | uob | 500 | 0.836 [0.802, 0.868] | 0.844 | 0.908 | 0.925 | 0.990 | 1.7 |
| molnextr | uspto | 489 | 0.789 [0.753, 0.826] | 0.804 | 0.804 | 0.906 | 0.980 | 1.2 |
| molnextr | jpo | 449 | 0.530 [0.481, 0.577] | 0.532 | 0.539 | 0.670 | 0.862 | 1.3 |
| molnextr | biovista_ocsr | 1000 | 0.403 [0.374, 0.432] | 0.455 | 0.458 | 0.731 | 0.956 | 1.7 |

## Exact match by ground-truth category

| tool | dataset | plain | chiral | markush |
|---|---|---|---|---|
| biominer_molglyph | clef | 0.875 (n=265) | 0.713 (n=171) | 0.000 (n=18) |
| biominer_molglyph | uob | 0.893 (n=496) | 0.250 (n=4) | - |
| biominer_molglyph | uspto | 0.846 (n=358) | 0.412 (n=131) | - |
| biominer_molglyph | jpo | 0.842 (n=386) | 0.586 (n=58) | 0.000 (n=5) |
| biominer_molglyph | biovista_ocsr | 0.867 (n=353) | 0.603 (n=179) | 0.739 (n=468) |
| molnextr | clef | 0.845 (n=265) | 0.713 (n=171) | 0.056 (n=18) |
| molnextr | uob | 0.839 (n=496) | 0.500 (n=4) | - |
| molnextr | uspto | 0.852 (n=358) | 0.618 (n=131) | - |
| molnextr | jpo | 0.541 (n=386) | 0.483 (n=58) | 0.200 (n=5) |
| molnextr | biovista_ocsr | 0.688 (n=353) | 0.564 (n=179) | 0.126 (n=468) |

## Paired tool comparison (exact match, A minus B)

| dataset | A | B | paired n | difference | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|---|
| clef | biominer_molglyph | molnextr | 454 | 0.015 | [-0.033, 0.062] | no |
| uob | biominer_molglyph | molnextr | 500 | 0.052 | [0.028, 0.076] | yes |
| uspto | biominer_molglyph | molnextr | 489 | -0.059 | [-0.110, -0.016] | yes |
| jpo | biominer_molglyph | molnextr | 449 | 0.269 | [0.220, 0.318] | yes |
| biovista_ocsr | biominer_molglyph | molnextr | 1000 | 0.357 | [0.323, 0.392] | yes |

## Failure types (counts)

| tool | dataset | correct | stereo_only | near_miss | wrong_structure | markush_mismatch | invalid_output | gt_unparseable |
|---|---|---|---|---|---|---|---|---|
| biominer_molglyph | biovista_ocsr | 760 | 47 | 9 | 60 | 119 | 5 | 0 |
| biominer_molglyph | clef | 354 | 30 | 9 | 43 | 18 | 0 | 46 |
| biominer_molglyph | jpo | 359 | 13 | 14 | 38 | 5 | 20 | 0 |
| biominer_molglyph | uob | 444 | 5 | 10 | 41 | 0 | 0 | 0 |
| biominer_molglyph | uspto | 357 | 41 | 10 | 73 | 0 | 8 | 11 |
| molnextr | biovista_ocsr | 403 | 13 | 34 | 112 | 394 | 44 | 0 |
| molnextr | clef | 347 | 32 | 3 | 49 | 17 | 6 | 46 |
| molnextr | jpo | 238 | 1 | 7 | 138 | 3 | 62 | 0 |
| molnextr | uob | 418 | 4 | 9 | 64 | 0 | 5 | 0 |
| molnextr | uspto | 386 | 7 | 8 | 78 | 0 | 10 | 11 |

Exact match = identical canonical isomeric SMILES. Unparseable predictions count as wrong. Categories: markush = wildcard/R-group atoms; chiral = has assigned stereo; plain = neither.
