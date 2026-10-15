# OCSR benchmark: smoke_test

- Finished: 2026-10-15T02:46:35+00:00 | config hash: 4f52697f9f09a941 | GPU: Tesla T4

## Overall (all scorable structures)

| tool | dataset | n | exact match [95% CI] | no-stereo | InChIKey conn. | mean Tanimoto | valid SMILES | img/s |
|---|---|---|---|---|---|---|---|---|
| biominer_molglyph | clef | 17 | 0.765 [0.529, 0.941] | 0.882 | 0.882 | 0.950 | 1.000 | 5.0 |
| biominer_molglyph | uob | 20 | 0.850 [0.699, 1.000] | 0.900 | 0.950 | 0.967 | 1.000 | 11.2 |
| biominer_molglyph | uspto | 20 | 0.800 [0.600, 0.950] | 0.950 | 1.000 | 0.993 | 1.000 | 6.9 |
| biominer_molglyph | jpo | 20 | 0.750 [0.550, 0.950] | 0.750 | 0.750 | 0.827 | 0.950 | 2.7 |
| biominer_molglyph | biovista_ocsr | 20 | 0.700 [0.500, 0.900] | 0.800 | 0.800 | 0.934 | 1.000 | 9.0 |
| molnextr | clef | 17 | 0.647 [0.412, 0.882] | 0.765 | 0.765 | 0.933 | 1.000 | 1.0 |
| molnextr | uob | 20 | 0.850 [0.699, 1.000] | 0.900 | 0.950 | 0.960 | 1.000 | 1.7 |
| molnextr | uspto | 20 | 0.850 [0.700, 1.000] | 0.900 | 0.900 | 0.958 | 1.000 | 1.1 |
| molnextr | jpo | 20 | 0.300 [0.149, 0.500] | 0.300 | 0.300 | 0.541 | 0.800 | 0.8 |
| molnextr | biovista_ocsr | 20 | 0.400 [0.200, 0.600] | 0.400 | 0.400 | 0.708 | 0.950 | 1.5 |

## Exact match by ground-truth category

| tool | dataset | plain | chiral | markush |
|---|---|---|---|---|
| biominer_molglyph | clef | 0.909 (n=11) | 0.600 (n=5) | 0.000 (n=1) |
| biominer_molglyph | uob | 0.850 (n=20) | - | - |
| biominer_molglyph | uspto | 1.000 (n=15) | 0.200 (n=5) | - |
| biominer_molglyph | jpo | 0.737 (n=19) | 1.000 (n=1) | - |
| biominer_molglyph | biovista_ocsr | 0.833 (n=6) | 0.333 (n=6) | 0.875 (n=8) |
| molnextr | clef | 0.727 (n=11) | 0.600 (n=5) | 0.000 (n=1) |
| molnextr | uob | 0.850 (n=20) | - | - |
| molnextr | uspto | 0.933 (n=15) | 0.600 (n=5) | - |
| molnextr | jpo | 0.316 (n=19) | 0.000 (n=1) | - |
| molnextr | biovista_ocsr | 0.667 (n=6) | 0.500 (n=6) | 0.125 (n=8) |

## Paired tool comparison (exact match, A minus B)

| dataset | A | B | paired n | difference | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|---|
| clef | biominer_molglyph | molnextr | 17 | 0.118 | [-0.176, 0.412] | no |
| uob | biominer_molglyph | molnextr | 20 | 0.000 | [-0.150, 0.150] | no |
| uspto | biominer_molglyph | molnextr | 20 | -0.050 | [-0.200, 0.100] | no |
| jpo | biominer_molglyph | molnextr | 20 | 0.450 | [0.250, 0.651] | yes |
| biovista_ocsr | biominer_molglyph | molnextr | 20 | 0.300 | [0.050, 0.550] | yes |

## Failure types (counts)

| tool | dataset | correct | stereo_only | near_miss | wrong_structure | markush_mismatch | invalid_output | gt_unparseable |
|---|---|---|---|---|---|---|---|---|
| biominer_molglyph | biovista_ocsr | 14 | 1 | 1 | 3 | 1 | 0 | 0 |
| biominer_molglyph | clef | 13 | 1 | 0 | 2 | 1 | 0 | 3 |
| biominer_molglyph | jpo | 15 | 0 | 0 | 4 | 0 | 1 | 0 |
| biominer_molglyph | uob | 17 | 1 | 0 | 2 | 0 | 0 | 0 |
| biominer_molglyph | uspto | 16 | 3 | 1 | 0 | 0 | 0 | 0 |
| molnextr | biovista_ocsr | 8 | 0 | 1 | 3 | 7 | 1 | 0 |
| molnextr | clef | 11 | 1 | 0 | 4 | 1 | 0 | 3 |
| molnextr | jpo | 6 | 0 | 0 | 10 | 0 | 4 | 0 |
| molnextr | uob | 17 | 1 | 0 | 2 | 0 | 0 | 0 |
| molnextr | uspto | 17 | 1 | 0 | 2 | 0 | 0 | 0 |

Exact match = identical canonical isomeric SMILES. Unparseable predictions count as wrong. Categories: markush = wildcard/R-group atoms; chiral = has assigned stereo; plain = neither.
