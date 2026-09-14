"""Populate measured results only after the complete independent audit passes."""
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
REPORT = EVIDENCE.parent/'mobility-repair-campaign-2026-09-14.md'


def main():
    audit = json.loads((EVIDENCE/'audit.json').read_text())
    rows = audit['comparisons']
    if not audit['passed'] or len(rows) != 6 or not all('current' in row for row in rows):
        raise ValueError('Final report requires all six originals/replays and native inspections')
    visual = json.loads((EVIDENCE/'visual_review.json').read_text())
    if {r['case'] for r in visual['cases']} != {f"{r['group']}_seed{r['seed']}" for r in rows}:
        raise ValueError('Manual contact/full-size review remains incomplete')
    views = sum(r['native_views_per_engine'] for r in rows)
    probes = sum(r['native_results']['unity']['collisionSamples'] for r in rows)
    prefix = EVIDENCE.name
    lines = ['## Measured results', '',
        f'All **six originals and six cold replays passed**. Every original passed both native engines: '
        f'**{views} planned views and {probes} floor/roof probe locations per engine**. '
        'Each probe location tests both floor and roof. Exact replay checks include semantic identities, '
        'raw mesh/GLB identities, collider bytes and collider inspection records.', '',
        f'The independent audit passed **{len(audit["checks"])} checks** and verified '
        f'**{audit["verified_artifact_receipts"]} generated artifact receipts**. '
        f'[Complete audit]({prefix}/audit.json)', '',
        '| Case | Visual triangles | Collider triangles | Reduction | Max sampled collider error | GLB |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['group'].title()} {r['seed']} | {r['current']['triangles']:,} | "
            f"{r['collider_output_triangles']:,} | {r['collider_reduction']:.0%} | "
            f"{r['collider_max_sampled_error_m']*1000:.2f} mm | {r['current']['asset_bytes']/1e6:.1f} MB |")
    lines += ['', f'![Measured visual and collision triangle counts]({prefix}/mesh_comparison.png)', '',
        'Comparison with the preceding textured campaign. Its colliders retained the full visual '
        'triangulation. Visual detail and required clearance remain independently checked; reducing '
        'collider triangles does not imply an equivalent reduction in rendered triangles.', '']
    precision_rows = [r for r in rows if r['precision_relaxed_vertices']]
    if precision_rows:
        lines += [f"{len(precision_rows)}/6 colliders needed local precision repair on "
            f"{min(r['precision_relaxed_vertices'] for r in precision_rows)}–"
            f"{max(r['precision_relaxed_vertices'] for r in precision_rows)} vertices each. "
            'The largest total vertex change was '
            f"{max(r['precision_maximum_vertex_change_m'] for r in precision_rows)*1000:.3f} mm. "
            'All then passed the full mesh/clearance/error checks and both native imports with '
            'the exact expected triangle counts. Per-case measurements remain in the audit.', '']
    lines += [
        '| Case | Upstream candidates | Accepted global relief scale | Local relief regions | Repaired required paths | Views per engine |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['group'].title()} {r['seed']} | {r['current']['recovery_attempts']} | "
            f"{r['global_relief_scale']:g} | {r['local_relief_regions']} | {r['repaired_path_count']} | {r['native_views_per_engine']} |")
    lines += ['', 'Relief scale zero means the procedural accretion layer was removed in the accepted '
        'candidate; it does not remove the underlying section shape or texture detail. Local masks '
        'can also reduce relief near defects. The raw repair journals preserve every rejected candidate. '
        'The multi-source seed-1 case uses the first deterministic replacement network in the same host; '
        'seed 42 is repaired on its original network. Source normal-map normalization ran in every case; '
        'no missing-source recovery is claimed by these naturally occurring cases.', '',
        '| Case | Original / cold replay | Peak worker RAM | Combined network length | Input profiles below eight voxels |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['group'].title()} {r['seed']} | {r['original_seconds']/60:.1f} / "
            f"{r['replay_seconds']/60:.1f} min | {r['peak_rss_mib']/1024:.2f} GiB | {r['current']['combined_length_m']:.1f} m | "
            f"{r['under_resolved_count']}/{r['resolution_section_count']} |")
    lines += ['', f"{sum(r['under_resolved_count'] > 0 for r in rows)}/6 cases retain input-profile "
        'resolution warnings. Those are not failed capsule checks: the continuous route tests apply '
        'to the actual triangles, while the eight-voxel screen asks whether finer geometric detail '
        'needs a separate convergence study. Passing this campaign does not resolve those warnings.', '']
    unity = [r['native_results']['unity']['editorPeakMemoryBytes']/2**30 for r in rows]
    unreal = [r['native_results']['unreal']['editor_peak_memory_bytes']/2**30 for r in rows]
    lines += ['', f'Recorded editor peak memory ranged from {min(unity):.2f}–{max(unity):.2f} GiB in Unity '
        f'and {min(unreal):.2f}–{max(unreal):.2f} GiB in Unreal. These include editor/import workloads; '
        'they are not minimum runtime requirements. Cooking, import and query timings remain in the '
        'native records and do not establish comparable engine frame rates.', '',
        f'![Predetermined middle floor views from both native engines]({prefix}/native_gallery.jpg)', '',
        'One predetermined middle floor-direction view per case and engine; views were not selected '
        'by appearance. Complete contact sheets and these full-size images were reviewed. Materials '
        'are visible and the inspected views show no obvious rectangular floor-projection seams. '
        'Unreal is substantially brighter; passing image guards does not establish matched appearance. '
        f'[Review scope and image identities]({prefix}/visual_review.json)', '',
        '### Inspect the generated assets', '',
        '| Case | Portable asset | Native Unity scene | Native Unreal project |',
        '|---|---|---|---|']
    for r in rows:
        root = '../../outputs/mobility_repair_campaign_20260914_v3'
        index = 0 if r['group']=='multi' else 1
        asset = f"{root}/seed_{r['seed']}/case_{index:04d}/attempt_0000/export/plume_cave.glb"
        native = f"{root}/native/{r['group']}_seed{r['seed']}"
        lines.append(f"| {r['group'].title()} {r['seed']} | [GLB]({asset}) | "
            f"[Scene]({native}/unity_project/Assets/PLUME_Inspection.unity) | "
            f"[Project]({native}/unreal_project/PLUMENative.uproject) |")
    lines += ['', 'Open each Unity scene within its corresponding `unity_project`. The native projects '
        'already contain the blended material and dedicated collider. The adjacent export package '
        'includes a Blender importer and material setup instructions. In Blender, import the GLB '
        '(or run `plume_cave_import_blender.py`), then run '
        '`continuous_material/apply_blender_material.py` to install the blended projection. '
        'These campaign exports are not saved `.blend` projects. No rocks were generated.', '']
    text = REPORT.read_text()
    start = '**Status: the fresh V3 campaign is running. Results below will be completed only\n' \
            'after generation, cold replay and native evidence have been audited.**'
    text = text.replace(start, '**Completed:** all six textured originals, all six cold replays and '
        'both native engines passed the frozen V3 protocol. Earlier failed revisions remain preserved.')
    if '## Measured results' in text:
        before, tail = text.split('## Measured results', 1)
        text = before+'## Automated regression coverage'+tail.split('## Automated regression coverage', 1)[1]
    text = text.replace('## Automated regression coverage', '\n'.join(lines)+'\n## Automated regression coverage')
    REPORT.write_text(text)
    output = EVIDENCE.parents[2]/'outputs/mobility_repair_campaign_20260914_v3'
    index = ['# Six inspected lava tubes', '',
        'Six distinct Earth designs with reusable 4K textures and no rocks: three single-source '
        'and three multi-source. Each has a separate cold replay for reproducibility; replay '
        'folders are not additional designs.', '',
        '[Measured results, repair history and limitations]'
        '(../../docs/reviews/mobility-repair-campaign-2026-09-14.md)', '',
        '| Design | Original package | Unity scene | Unreal project |',
        '|---|---|---|---|']
    for r in rows:
        case_index = 0 if r['group']=='multi' else 1
        package = f"seed_{r['seed']}/case_{case_index:04d}/attempt_0000/export"
        native = f"native/{r['group']}_seed{r['seed']}"
        index.append(f"| {r['group'].title()} {r['seed']} | [GLB]({package}/plume_cave.glb) | "
            f"[Scene]({native}/unity_project/Assets/PLUME_Inspection.unity) | "
            f"[Project]({native}/unreal_project/PLUMENative.uproject) |")
    index += ['', 'Open Unity scenes within their corresponding `unity_project` folder. The native '
        'projects contain the blended rock material and separately checked collider.', '',
        'For Blender, import a GLB (or run its adjacent `plume_cave_import_blender.py`), then '
        'run `continuous_material/apply_blender_material.py` from the same export folder. '
        'Save your inspection as a new `.blend` file. Ordinary GLB import alone uses the portable '
        'UV material and does not install the blended projection.', '',
        'Required dominant routes passed the configured 0.5 m high × 0.5 m wide upright-body '
        'check with 0.02 m margin. Optional branches may remain narrower. This does not establish '
        'vehicle dynamics, ground contact or geological realism.', '']
    (output/'README.md').write_text('\n'.join(index))
    print('Final report populated from the completed audit')


if __name__ == '__main__':
    main()
