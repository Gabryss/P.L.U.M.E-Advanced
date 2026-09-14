# Texture repair validation — 13 September 2026

Texture inspection and bounded repair now run automatically during normal full
generation, target export and full reliability campaigns. UV-collapse repair and
the displacement/geometry acceptance loop already existed; this change adds map
inspection, recorded normal repair and a texture-package rebuild budget.

The production source and shipped material-resource fingerprint tested here is
`60074a71b013236cbec71132ca08d4d736187e313d0a1361d677592c3b3116a7`.
No production code or source assets changed during the final campaign.

## Behavior

- Prepare shared portable color, normal and roughness PNGs, preserving original
  source files, image aspect ratios and the configured resolution cap. Preserve
  dark 16-bit roughness levels; leave displacement precision to the geometry pass.
- Inspect normal vectors after resizing. Normalize usable vectors once, with an
  explicit green-channel conversion only when `cave_normal_convention="directx"`.
  Undefined normals, missing files and corrupt sources are rejected.
- Check exported GLB pixels, material bindings, repeat samplers, normal strength
  and roughness channel packing against accepted maps. Check OBJ references and
  continuous-material settings, images and Blender/Unity/Unreal adapter files.
- Rebuild a failed texture package once from the accepted scene. The network,
  mesh, seed and acceptance limits do not change. Reinspect before publication;
  on exhaustion retain the previous published package and failure diagnostics.
- Include shader resources in the version fingerprint and the texture journal
  in cold-replay identity checks. Run completion verifies that inspected files
  and the journal are still intact.

See [configuration and scope](../materials.md#automatic-texture-inspection-and-repair).
The default `geometry.texture_repair_attempts=1` requires no extra command.

## Validation results

| Check | Result |
| --- | --- |
| Broad regression suite | 763 passed; 59 additional subtests passed |
| Dedicated texture tests, included above | 40 passed |
| Integration tests | 3 passed: real stamping repair, CLI recovery/resume, native Blender projection |
| Optional checks | 18 skipped in the broad suite: 17 external shader compiler checks and one optional collection |
| Static checks | Ruff passed; Mypy passed for 104 production files |
| Full textured generation | Passed; Earth, root seed 17, one network, 80 m target, 12 cm voxels, no rocks |
| Cold replay | Passed; Python hash seeds 11 and 37; identical host, network, sections, mesh, GLB and both recovery journals |
| Portable asset checks | 41/41 passed in each run |
| Stored evidence audit | 86 artifact hashes verified; report-only campaign audit passed |

The regression selection excluded integration, performance and paper markers;
the three integration tests above ran separately. The 40 texture tests cover
missing/corrupt/truncated files, undefined and grayscale normals, explicit normal
conventions, downsampling, 16-bit roughness, partial materials, shared event maps,
binding and sampler corruption, stale/missing native maps and shader files,
incorrect settings, retry exhaustion, resource failures, source preservation,
post-export tampering, deterministic repair journals and shader fingerprints.

The real rock normal tile contained 16,777,216 pixels after conversion to 4K:

| Normal measurement | Before repair | After repair |
| --- | ---: | ---: |
| Pixels with length error greater than 0.02 | 472,092 | 0 |
| Undefined vectors | 0 | 0 |
| Maximum length error | 0.819310 | 0.006650 |

The production case accepted the first prepared package and needed one normal-map
repair. Package-rebuild success and exhaustion were exercised by fault tests.
The generation contained 208,788 triangles across 80.32 m of route. Its GLB is
71.64 MB; the complete package is about 213.9 MiB including alternative formats,
shared maps and native material files. Peak worker memory was 1,140 MiB or less.
Collision simplification retained the full collider after inspection rejected the
simplified candidate. These are material integration results, not evidence of
network diversity or guaranteed continuous traversability.

Blender **4.0.1** passed 27 native normal-orientation cases, UV-invariance and image
reuse checks. Unity and Unreal bundle contents were checked; their native editors
and shader compilers were not run. The user's Blender 5.2 environment was not
available in this validation environment.

## Evidence and reproduction

Small evidence files remain in the repository even if `outputs/` is cleared:

- [Verification summary and identities](texture-evidence-2026-09-13/verification.json)
- [Actual texture repair journal](texture-evidence-2026-09-13/texture_recovery.json)
- [Native Blender measurements](texture-evidence-2026-09-13/blender_material_check.json)
- [Regression results](texture-evidence-2026-09-13/regression.xml) and [integration results](texture-evidence-2026-09-13/integration.xml)
- [Generation recipe](texture-evidence-2026-09-13/single_80m_4k.toml)

The [working campaign report](../../outputs/texture_recovery_20260913/final/campaign/report.html)
links the generation and cold replay. Earlier interrupted work is kept separately
under `outputs/texture_recovery_20260913/`; only `final/` is passing evidence.

From the repository root, with its dependencies and rock source images installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/plume-mpl \
  .venv/bin/plume-check \
  --configs docs/reviews/texture-evidence-2026-09-13/single_80m_4k.toml \
  --seeds 17 --scope full --timeout 1800 --memory-limit-mib 8192 \
  --output outputs/my_texture_repair_campaign
```

A correct directional tile can still show UV-chart seams when imported as a
standard GLB material. Use the supplied continuous projection material in the
native application. Numerical texture acceptance does not certify lighting,
tile seamlessness, geological appearance or every possible input image.
