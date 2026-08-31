# Frozen PDC cave-level partition

These files partition all 95 cave identifiers in the audited Pyroduct Digital
Catalog v2.0 tree (`sha256:885b61443867710d6759558767e67d61399c1964da58d17d0dc56a52ff5959bb`).
No individual cave appears in both files.

The split was frozen before morphology tuning. Cave identifiers were ranked by
the hexadecimal SHA-256 digest of `pdc-v2.0:20260831:<cave-id>`; the first 19
ranked identifiers form the untouched 20% evaluation partition and the other
76 form the calibration partition. No morphometric value was used to choose or
move a cave between partitions.

Parameter selection may use only `pdc_calibration_caves.txt`. An exploratory
whole-catalog summary was produced before this split existed, so the evaluation
partition is not claimed to be historically unseen. From this freeze onward it
is locked out of tuning code and serves as the confirmatory partition. It is
used once after the model and configuration are frozen. Resplitting requires a
new declared experiment, not an edit to these files.
