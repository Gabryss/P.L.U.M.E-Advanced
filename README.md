# P.L.U.M.E-Advanced

P.L.U.M.E. Advanced is a voxel-based cave generation prototype built from the project specifications in [`meeting_outcome`](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/meeting_outcome).

The implementation follows the required causal pipeline:

`geology -> hydro -> dissolution -> void formation -> mesh extraction -> dataset export`

## Structure

- `src/geology`: layered material and fracture-field generation
- `src/hydro`: reduced hydraulic-head solver and flow estimation
- `src/speleogenesis`: dissolution, porosity, permeability feedback, void formation
- `src/meshing`: mesh extraction and OBJ export
- `src/dataset`: batch generation entrypoint
- `src/visualization`: metrics, plots, slices, mesh previews, progress helpers
- `config/default.yaml`: reproducible default configuration
- `tests/`: deterministic unit and integration coverage

## Run

Single run:

```bash
python3 -m src.cli run --config config/default.yaml
```

Batch run:

```bash
python3 -m src.cli batch --config config/default.yaml --samples 3
```

Outputs are written under `outputs/` by default and include:

- `config_snapshot.yaml`
- `run.log`
- `metrics.csv`
- `debug/` plots and slice images
- `mesh/cave.obj`
- `summary.json`

## Notes

- `tqdm` is optional. If unavailable, the project falls back to simple console progress messages.
- `scikit-image` is optional. If unavailable, meshing falls back to voxel-surface extraction instead of marching cubes.
