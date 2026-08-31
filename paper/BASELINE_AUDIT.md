# Baseline audit

Baseline revision: `bc0afdb88d460e392f9a49fa63da694466ee3657` on
`feature/major-procedural-upgrade`. The tree was clean before this work.

1. Stage A used exactly the documented 0.12/0.10/0.22/0.28/0.28 weights.
2. Network steering consumes the combined routing cost. That scalar also
   conditions passage morphology, not a second hidden route-cost term.
3. Stage-C controls use segment state, arc length, and a seed-derived segment
   phase.
4. Adaptive sample count did not change the number of RNG draws per segment.
5. Dense/tiled and seam behavior already had regression coverage; paper
   measurements remain required.
6. Chunk meshes are globally assembled and welded.
7. Named stage sub-seeds isolate A-C from event configuration.
8. Export adapters run after canonical geometry and cannot alter A-C.
9. UV charting, metric scaling, normals/tangents, PBR packaging, smoothing, and
   visual displacement exist. LODs, wall shells, and geology-conditioned
   material synthesis remain deferred.
10. The old report lacked a stable full semantic graph; one is now emitted.
11. Public generators can stop after A/B/C.
12. Geometry can be benchmarked without visual-surface packaging.
13. Body and flow-regime inputs resolve separately.
14. Morphology defaults are heuristics unless later calibrated at cave level.
15. Junction and segment IDs support direct split/merge/chamber metrics.
