# Ethics

`community-analyzer` is built to detect **structural distortion** in
organizational / community communication — not to evaluate individuals．

See also：[Architecture](architecture.md) ／ [Use cases](use-cases.md)．

## Intended use

* This library is for detecting structural distortion in organizational /
  community communication．It **must not** be repurposed for individual
  evaluation, ranking, or punishment．
* The hypotheses describe patterns of communication health at the level of
  channels, threads, and roles — not the worth or performance of any one
  person．The summary thresholds are first-pass calibration heuristics, not
  clinical or HR judgements．

## Aggregate-only output

* Reports show **aggregate values only**．
* The only per-user artifact is `tables/h3_pagerank.csv`，and it is
  **anonymized by default**．

## Anonymization

`Anonymizer` (constructed with `Anonymizer::new(true)`) replaces raw user IDs
with salted hashes when writing output．The salt is a **random per-run value
that is never persisted**，so anonymized IDs cannot be correlated across runs．
Anonymization can be disabled explicitly by the caller，but it is on by
default．

## No-raw-ID-leak audit

A runtime audit test, `test_finalize_report_assets_no_raw_id_leak`, enforces
that no raw platform user ID may appear in `report.md`, `metrics.json`, the
figures, or the PageRank CSV unless anonymization is explicitly disabled．This
guarantee is part of the test suite, so the no-leak contract is checked on
every run．
