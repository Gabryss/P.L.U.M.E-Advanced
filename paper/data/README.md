# Pyroduct Digital Catalog input

The paper evaluation uses the **Pyroduct Digital Catalog v2.0**, Zenodo record
`17750755`, DOI `10.5281/zenodo.17750755`. The source archive is external
research data and must not be committed to this repository.

Zenodo lists the v2.0 ZIP as 82.4 MB with MD5
`cccad95bbf3ef56d1bd48ac75273682c`. The record's requested dataset citation
uses the all-versions DOI `10.5281/zenodo.14535885`:

> Romio, F. A. P., Lobosco, G., Marraffa, A., Pisani, L., & Tomasi, I. (2025).
> Pyroduct Digital Catalog (2.0) [Data set]. Zenodo.

1. Download version 2.0 from the Zenodo record and verify the checksum shown by
   Zenodo for the downloaded archive.
2. Extract it outside this Git checkout.
3. Set `PLUME_PDC_ROOT` to the extracted directory containing the recursive
   TXT cross-section collection.
4. Run `uv run plume-evaluate --config paper/experiments.toml pdc-audit`.
5. Inspect `paper/outputs/pdc_audit.json`, the inventory, and every rejection
   before running morphometry.

The loader never modifies source data. It retains relative paths, raw points,
cave IDs, section IDs, and explicit rejection reasons. If catalog filenames do
not encode cave identity reliably, reconcile the catalog spreadsheet metadata
before the final experiment freeze. Retrieve the exact BibTeX from Zenodo when
assembling the manuscript bibliography.
