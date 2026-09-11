# PLUME-Advanced: illustrated manuscript and evaluation results

This corrected package includes all ten figures already inserted in `main.tex`. No figure downloads, renaming, or manual copying are needed. The compiled review copy is `PLUME_Advanced_With_Figures.pdf`.

The retained event illustration enables Rocky props on the current Earth seed-4 scenario: 641 rocks and eight boulders are generated as actual meshes. Figure 6 now includes an interior render and measured rock sizes; Figure 7 refreshes the associated atlas labels. Figures 2–4 retain the removal of redundant overall titles and subtitles. Captions, saved figure data, and provenance are updated together.

## Open in Overleaf

1. Choose **New Project → Upload Project** and upload `PLUME_Advanced_Overleaf_With_Figures.zip`.
2. If Overleaf asks for the main document, choose **main.tex**. Use **pdfLaTeX**.
3. Recompile. All ten figures appear automatically.

A new project avoids mixing this revision with the earlier placeholder version. The corrected manuscript includes images directly, so a missing figure file produces a compilation error instead of an empty figure box.

## Included files

- `main.tex`: detailed IEEE conference-format manuscript, with no page limit imposed.
- `references.tex`: bibliography included directly; no BibTeX step is required.
- `PLUME_Advanced_With_Figures.pdf`: compiled illustrated review copy.
- `figures/`: the ten active images plus editable source for the pipeline schematic.
- `FIGURE_GUIDE.md`: figure contents, source information, and later research figures to prepare.
- `REVIEW_NOTES.md`: scientific corrections and remaining evidence requirements.
- `provenance.json`: review context, file hashes, and inserted-figure mappings.
- `figure_candidates/`: earlier documentation images retained as reference material; no action is needed with these files.
- `source_snapshot/`: the original manuscript, configuration records, and data needed to replot the current event/atlas figures in an installed PLUME environment; this is not a runnable relocated generator.

The full 7 September 2026 campaign attempted 2,076 cases: 2,071 completed and five dense 5 km cases reached the memory limit. Tables IV–VI now contain measured results.
Figure 9 compares the morphology distributions; Figure 10 shows every resource
attempt, including the five dense 5 km runs stopped at the 12 GiB budget. The
remaining figures retain their labelled schematic or development scope.

The bundled `evaluation/` directory contains the campaign report, source
snapshots, verified configuration identities and compact measurement tables.
See `evaluation/PACKAGE_SCOPE.md` for what is included and how original paths
relate to the repository. The external PDC source dataset, texture assets and
large application packages are not required to compile the PDF.

The results retain negative findings. Shape improvements relative to matched
ellipses do not remove the remaining size-distribution mismatch; lower memory
use does not establish final-mesh fidelity. Adaptive sampling had greater point-set discrepancy in all 30 comparisons, and distributary tendency did not increase median cycle rank. All 33 A–C repeatability checks and 15 package-presence checks passed; application imports and simulator tests remain unperformed.

Compile locally with `latexmk -pdf main.tex`. The pipeline PDF is already bundled; main-document compilation does not need to compile `figures/f01_pipeline.tex` separately.
