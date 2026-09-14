# Textured campaign execution

Campaign root: `outputs/textured_campaign_20260913/`. Started 13 September and continued into 14 September 2026 (Europe/Paris); paths retain the start date.

`protocol_v2.json` is the final six-design cohort. All runs use Earth, a 250 m target, fixed 0.12 m voxels, three 4096² PBR maps, a 4 m reusable tile and normal strength 1. Rocks/events and baked height-map displacement are disabled. Single-source and three-source groups each use seeds 1, 42 and 4294967295. The target is not summed branch length.

The initial `protocol.json` shortened the single-source recipe to 150 m without reducing its required loops/side branches. All three seeds rejected that configuration before meshing. Their results remain in `single/`. The final single-source recipe restores the existing 250 m target and retains all three seeds. The multi-source recipe is unchanged. Initial failures are separate from the six-design final cohort.

Two independent campaign groups run concurrently, with one worker per group. Each successful original is repeated in a new process with no checkpoint reuse; Python hash seeds are 11 and 37. Per-worker limits are 3600 seconds and 8192 MiB of address space. These are bounded evaluation runs, not isolated speed benchmarks.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/plume-check \
  --configs docs/reviews/textured-campaign-evidence-2026-09-13/single_250m_4k.toml \
  --seeds 1 42 4294967295 --scope full --timeout 3600 --memory-limit-mib 8192 \
  --output outputs/textured_campaign_20260913/single_250

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/plume-check \
  --configs docs/reviews/textured-campaign-evidence-2026-09-13/multi_250m_4k.toml \
  --seeds 1 42 4294967295 --scope full --timeout 3600 --memory-limit-mib 8192 \
  --output outputs/textured_campaign_20260913/multi
```

For each accepted original, `scripts/check_native_engines.py` uses isolated Unity 6000.6.0f1 / URP 17.6.0 / glTFast 6.20.0 and Unreal 5.8.2 projects. Unity and Unreal are run sequentially to share the GPU. Native inspection imports the exact approved GLB, applies the shipped continuous projection material, verifies image resolution/color spaces, checks every saved floor/roof sample against the final visual export, renders two interior views and tests normal-map influence. Unity additionally checks invariance to deliberately corrupted UV coordinates. Unreal's validation mesh disables the automatically enabled Nanite representation and uses the complete visual mesh for comparison. This is not a test of a production collision simplification or Nanite fallback.

```sh
.venv/bin/python scripts/check_native_engines.py ORIGINAL_ATTEMPT_DIRECTORY \
  --output NEW_NATIVE_DIRECTORY \
  --unity /home/gabriel/Unity/Hub/Editor/6000.6.0f1/Editor/Unity \
  --unreal /home/gabriel/Downloads/Linux_Unreal_Engine_5.8.2/Engine/Binaries/Linux/UnrealEditor \
  --timeout 1800
```

Optional material tests use Blender 4.0.1 on the CPU and DXC with the installed Unity package headers. The Blender fixture checks 27 normal cases and UV/color-channel behavior; this is shader-fixture coverage, not six additional full Blender inspections. The two shared-HLSL glslang cases remain unavailable if that compiler is absent. Full native Vulkan compilation is checked separately in the actual editors.

The source fingerprint, recipe hashes, complete case plan and dependency versions are frozen in the protocol. The read-only audit verifies retained results, artifacts, replay identities, repair budgets and native receipts. Reruns require a new output directory when inputs or source change. Do not replace failed seeds in-place.

The first native trial accepted partially overexposed views. `native_protocol_v2.json` records the subsequent inspection-only change: light at the verified passage centre, at most eight half-intensity trials, a 0.5% clipping target and a 1% final rejection threshold. Both engines were rerun for every accepted original under this revision, without changing the generated inputs. `native_campaign_v1.json` and the original trial projects remain as evidence.
