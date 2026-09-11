"""Portable GLB validation with human and machine-readable progress reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np
import trimesh
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

ValidationProgress = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ValidationCheck:
    """One portable-asset validation result."""

    category: str
    name: str
    passed: bool
    detail: str


class GlbAsset:
    """Parsed JSON and binary chunks from one GLB 2.0 asset."""

    _COMPONENT_DTYPES: ClassVar[dict[int, np.dtype]] = {
        5120: np.dtype("<i1"),
        5121: np.dtype("<u1"),
        5122: np.dtype("<i2"),
        5123: np.dtype("<u2"),
        5125: np.dtype("<u4"),
        5126: np.dtype("<f4"),
    }
    _COMPONENT_COUNTS: ClassVar[dict[str, int]] = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self.path.read_bytes()
        if len(self.data) < 28 or self.data[:4] != b"glTF":
            raise ValueError("Asset is not a binary glTF file")
        self.version = struct.unpack_from("<I", self.data, 4)[0]
        self.declared_length = struct.unpack_from("<I", self.data, 8)[0]
        json_length, json_type = struct.unpack_from("<II", self.data, 12)
        if json_type != 0x4E4F534A:
            raise ValueError("First GLB chunk is not JSON")
        json_start = 20
        json_end = json_start + json_length
        self.document = json.loads(self.data[json_start:json_end].rstrip(b" "))
        binary_header = json_end
        binary_length, binary_type = struct.unpack_from("<II", self.data, binary_header)
        if binary_type != 0x004E4942:
            raise ValueError("Second GLB chunk is not binary data")
        binary_start = binary_header + 8
        self.binary = memoryview(self.data)[binary_start : binary_start + binary_length]

    def accessor(self, accessor_index: int) -> np.ndarray:
        accessor = self.document["accessors"][accessor_index]
        view = self.document["bufferViews"][accessor["bufferView"]]
        dtype = self._COMPONENT_DTYPES[accessor["componentType"]]
        components = self._COMPONENT_COUNTS[accessor["type"]]
        count = int(accessor["count"])
        offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        stride = int(view.get("byteStride", dtype.itemsize * components))
        values = np.ndarray(
            shape=(count, components),
            dtype=dtype,
            buffer=self.binary,
            offset=offset,
            strides=(stride, dtype.itemsize),
        ).copy()
        return values[:, 0] if accessor["type"] == "SCALAR" else values

    def embedded_image(self, image_index: int):
        from PIL import Image

        image = self.document["images"][image_index]
        view = self.document["bufferViews"][image["bufferView"]]
        start = int(view.get("byteOffset", 0))
        end = start + int(view["byteLength"])
        return Image.open(BytesIO(bytes(self.binary[start:end]))).copy()

    def cave_primitive(self) -> tuple[dict, dict, dict]:
        node = next(
            node
            for node in self.document["nodes"]
            if node.get("name") == "cave_wall"
        )
        mesh = self.document["meshes"][node["mesh"]]
        return node, mesh, mesh["primitives"][0]


class PortableAssetValidator:
    """Run simulator-neutral checks against the complete exported scene."""

    PHASES = (
        "GLB container",
        "Embedded material maps",
        "Geometry and topology",
        "Normals and tangents",
        "UV continuity",
        "Scene completeness",
        "Baked displacement",
        "Reproducibility",
    )

    def __init__(
        self,
        asset_path: str | Path,
        *,
        manifest_path: str | Path | None = None,
        run_manifest_path: str | Path | None = None,
    ) -> None:
        self.asset_path = Path(asset_path).expanduser().resolve()
        self.glb = GlbAsset(self.asset_path)
        candidate_manifest = (
            Path(manifest_path)
            if manifest_path is not None
            else self.asset_path.with_suffix(".manifest.json")
        )
        self.manifest = (
            json.loads(candidate_manifest.read_text(encoding="utf-8"))
            if candidate_manifest.is_file()
            else {}
        )
        candidate_run_manifest = (
            Path(run_manifest_path)
            if run_manifest_path is not None
            else next(
                (
                    parent / "run_manifest.json"
                    for parent in self.asset_path.parents
                    if (parent / "run_manifest.json").is_file()
                ),
                self.asset_path.parents[1] / "run_manifest.json",
            )
        )
        self.run_manifest_path = candidate_run_manifest
        self.run_manifest = (
            json.loads(candidate_run_manifest.read_text(encoding="utf-8"))
            if candidate_run_manifest.is_file()
            else {}
        )

    def validate(
        self,
        *,
        progress: ValidationProgress | None = None,
    ) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        phase_methods = (
            self._container_checks,
            self._material_checks,
            self._geometry_checks,
            self._normal_tangent_checks,
            self._uv_checks,
            self._scene_checks,
            self._displacement_checks,
            self._reproducibility_checks,
        )
        for index, (phase, method) in enumerate(
            zip(self.PHASES, phase_methods, strict=True),
            start=1,
        ):
            checks.extend(method())
            if progress is not None:
                progress(index, len(self.PHASES), phase)
        return checks

    @staticmethod
    def _check(
        category: str,
        name: str,
        condition: bool,
        detail: str,
    ) -> ValidationCheck:
        return ValidationCheck(category, name, bool(condition), detail)

    def _container_checks(self) -> list[ValidationCheck]:
        document = self.glb.document
        external_buffers = [
            buffer["uri"] for buffer in document.get("buffers", ()) if "uri" in buffer
        ]
        external_images = [
            image["uri"] for image in document.get("images", ()) if "uri" in image
        ]
        return [
            self._check(
                "container",
                "GLB version",
                self.glb.version == 2,
                f"version={self.glb.version}",
            ),
            self._check(
                "container",
                "Declared byte length",
                self.glb.declared_length == len(self.glb.data),
                f"declared={self.glb.declared_length}, actual={len(self.glb.data)}",
            ),
            self._check(
                "container",
                "Self-contained buffers",
                not external_buffers,
                f"external_buffers={external_buffers}",
            ),
            self._check(
                "container",
                "Self-contained images",
                not external_images,
                f"external_images={external_images}",
            ),
        ]

    def _material_checks(self) -> list[ValidationCheck]:
        document = self.glb.document
        node, _mesh, primitive = self.glb.cave_primitive()
        material = document["materials"][primitive["material"]]
        pbr = material.get("pbrMetallicRoughness", {})
        decoded: list[str] = []
        valid_images = True
        for index, image in enumerate(document.get("images", ())):
            try:
                decoded_image = self.glb.embedded_image(index)
                decoded.append(
                    f"{index}:{decoded_image.width}x{decoded_image.height}:{image.get('mimeType')}"
                )
                valid_images &= decoded_image.width > 0 and decoded_image.height > 0
            except Exception as error:
                decoded.append(f"{index}:{type(error).__name__}")
                valid_images = False
        normal_scale = float(material.get("normalTexture", {}).get("scale", 1.0))
        return [
            self._check(
                "materials",
                "Embedded images decode",
                valid_images and bool(decoded),
                ", ".join(decoded),
            ),
            self._check(
                "materials",
                "Base-color texture bound",
                "baseColorTexture" in pbr,
                f"material={material.get('name')}",
            ),
            self._check(
                "materials",
                "Metallic-roughness texture bound",
                "metallicRoughnessTexture" in pbr,
                f"material={material.get('name')}",
            ),
            self._check(
                "materials",
                "Normal texture bound",
                (
                    "normalTexture" in material
                    and math.isfinite(normal_scale)
                    and normal_scale >= 0.0
                ),
                f"normal_scale={normal_scale}",
            ),
            self._check(
                "materials",
                "Interior-only cave material",
                material.get("doubleSided") is False,
                f"doubleSided={material.get('doubleSided')}",
            ),
            self._check(
                "materials",
                "Cave metadata present",
                bool(node.get("extras")),
                f"extras={sorted(node.get('extras', {}))}",
            ),
        ]

    def _geometry_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _node, _mesh, primitive = self.glb.cave_primitive()
        attributes = primitive["attributes"]
        positions = np.asarray(
            self.glb.accessor(attributes["POSITION"]),
            dtype=np.float64,
        )
        normals = np.asarray(
            self.glb.accessor(attributes["NORMAL"]),
            dtype=np.float64,
        )
        tangents = np.asarray(
            self.glb.accessor(attributes["TANGENT"]),
            dtype=np.float64,
        )
        texcoords = np.asarray(
            self.glb.accessor(attributes["TEXCOORD_0"]),
            dtype=np.float64,
        )
        faces = np.asarray(
            self.glb.accessor(primitive["indices"]),
            dtype=np.int64,
        ).reshape((-1, 3))
        return positions, faces, normals, tangents, texcoords

    def _geometry_checks(self) -> list[ValidationCheck]:
        positions, faces, _normals, _tangents, _texcoords = self._geometry_arrays()
        index_valid = bool(
            faces.size
            and int(faces.min()) >= 0
            and int(faces.max()) < len(positions)
        )
        triangles = positions[faces]
        double_areas = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        loaded = trimesh.load(self.asset_path, force="scene", process=False)
        scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
        cave_key = next(key for key in scene.geometry if "cave_wall" in key)
        cave_mesh = scene.geometry[cave_key]
        # UV chart seams duplicate positions legitimately. Check geometric
        # topology after exact positional welding, without hiding real cracks.
        _, geometric_indices = np.unique(positions, axis=0, return_inverse=True)
        geometric_faces = geometric_indices[faces]
        edges = np.sort(
            np.concatenate(
                [
                    geometric_faces[:, [0, 1]],
                    geometric_faces[:, [1, 2]],
                    geometric_faces[:, [2, 0]],
                ]
            ),
            axis=1,
        )
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = int(np.count_nonzero(edge_counts == 1))
        nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
        collision_path = self.asset_path.with_name(f"{self.asset_path.stem}_collision.obj")
        collision_required = (
            getattr(self, "run_manifest", {}).get("resolved_config", {})
            .get("export", {}).get("generate_collision", True) is not False
        )
        collision_valid = not collision_required
        collision_detail = (
            f"missing={collision_path}" if collision_required
            else "disabled in resolved export configuration"
        )
        if collision_path.is_file():
            collision = trimesh.load_mesh(collision_path, process=False)
            collision_valid = bool(
                len(collision.faces) > 0
                and np.isfinite(collision.vertices).all()
                and collision.is_winding_consistent
                and collision.is_watertight
            )
            collision_detail = (
                f"vertices={len(collision.vertices)}, faces={len(collision.faces)}, "
                f"winding_consistent={collision.is_winding_consistent}, "
                f"watertight={collision.is_watertight}"
            )
        summary = self.manifest.get("summary", {})
        voxel_components = int(summary.get("voxel_component_count", 0))
        return [
            self._check(
                "geometry",
                "Closed manifold cave wall",
                boundary_edges == 0 and nonmanifold_edges == 0,
                f"boundary_edges={boundary_edges}, nonmanifold_edges={nonmanifold_edges}",
            ),
            self._check(
                "geometry",
                "Finite vertex positions",
                bool(np.isfinite(positions).all()),
                f"vertices={len(positions)}",
            ),
            self._check(
                "geometry",
                "Triangle indices in bounds",
                index_valid,
                f"faces={len(faces)}",
            ),
            self._check(
                "geometry",
                "No degenerate faces",
                bool(np.all(double_areas > 1e-10)),
                f"minimum_double_area={float(double_areas.min()):.6g}",
            ),
            self._check(
                "geometry",
                "Consistent winding",
                bool(cave_mesh.is_winding_consistent),
                f"winding_consistent={cave_mesh.is_winding_consistent}",
            ),
            self._check(
                "geometry",
                "Connected traversable voxel volume",
                voxel_components in {0, 1},
                f"voxel_component_count={voxel_components}",
            ),
            self._check(
                "geometry",
                "Conservative collision sidecar",
                collision_valid,
                collision_detail,
            ),
        ]

    def _normal_tangent_checks(self) -> list[ValidationCheck]:
        _positions, _faces, normals, tangents, _texcoords = self._geometry_arrays()
        normal_error = np.abs(np.linalg.norm(normals, axis=1) - 1.0)
        tangent_error = np.abs(np.linalg.norm(tangents[:, :3], axis=1) - 1.0)
        orthogonality = np.abs(np.einsum("ij,ij->i", normals, tangents[:, :3]))
        handedness_error = np.abs(np.abs(tangents[:, 3]) - 1.0)
        return [
            self._check(
                "normals",
                "Unit geometric normals",
                bool(np.max(normal_error) <= 5e-4),
                f"maximum_error={float(np.max(normal_error)):.6g}",
            ),
            self._check(
                "normals",
                "Unit tangents",
                bool(np.max(tangent_error) <= 5e-4),
                f"maximum_error={float(np.max(tangent_error)):.6g}",
            ),
            self._check(
                "normals",
                "Normal-tangent orthogonality",
                bool(np.max(orthogonality) <= 5e-4),
                f"maximum_dot={float(np.max(orthogonality)):.6g}",
            ),
            self._check(
                "normals",
                "Tangent handedness",
                bool(np.max(handedness_error) <= 1e-6),
                f"maximum_error={float(np.max(handedness_error)):.6g}",
            ),
        ]

    def _uv_checks(self) -> list[ValidationCheck]:
        positions, faces, normals, tangents, texcoords = self._geometry_arrays()
        spans = np.ptp(texcoords[faces], axis=1)
        excessive = np.any(spans > 0.500001, axis=1)
        excessive_count = int(np.count_nonzero(excessive))
        excessive_fraction = float(np.mean(excessive))
        excessive_limit = max(2, int(math.ceil(0.0005 * len(faces))))

        rounded = np.round(positions, decimals=6)
        _unique, inverse, counts = np.unique(
            rounded,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        integer_error = 0.0
        maximum_tangent_angle = 0.0
        chart_seam_groups = 0
        chart_seam_vertices = 0
        for group_index in np.flatnonzero(counts > 1):
            group = np.flatnonzero(inverse == group_index)
            uv_delta = texcoords[group] - texcoords[group[0]]
            group_integer_error = float(
                np.max(np.abs(uv_delta - np.round(uv_delta)))
            )
            if group_integer_error > 1e-4:
                chart_seam_groups += 1
                chart_seam_vertices += len(group)
                continue
            integer_error = max(integer_error, group_integer_error)
            dots = np.clip(
                tangents[group, :3] @ tangents[group[0], :3],
                -1.0,
                1.0,
            )
            maximum_tangent_angle = max(
                maximum_tangent_angle,
                float(np.max(np.degrees(np.arccos(dots)))),
            )
        uv_scale_m = float(
            self.manifest.get("cave", {})
            .get("material", {})
            .get("uv_scale_m", 8.0)
        )
        world_edges = np.stack(
            (
                positions[faces[:, 1]] - positions[faces[:, 0]],
                positions[faces[:, 2]] - positions[faces[:, 0]],
            ),
            axis=2,
        )
        uv_edges = np.stack(
            (
                texcoords[faces[:, 1]] - texcoords[faces[:, 0]],
                texcoords[faces[:, 2]] - texcoords[faces[:, 0]],
            ),
            axis=2,
        )
        uv_determinants = np.abs(np.linalg.det(uv_edges))
        valid_metric = uv_determinants > 1e-12
        collapsed_uv_count = int(np.count_nonzero(~valid_metric))
        singular_values = np.linalg.svd(
            world_edges[valid_metric] @ np.linalg.inv(uv_edges[valid_metric]),
            compute_uv=False,
        )
        metric_scale = (
            np.sqrt(singular_values[:, 0] * singular_values[:, 1])
            / max(uv_scale_m, 1e-9)
        )
        anisotropy = singular_values[:, 0] / np.maximum(
            singular_values[:, 1],
            1e-12,
        )
        vertical_normal = normals[faces].mean(axis=1)[:, 1][valid_metric]
        roof = vertical_normal < -0.55
        floor = vertical_normal > 0.55
        enough_directional_faces = (
            int(np.count_nonzero(roof)) >= 100
            and int(np.count_nonzero(floor)) >= 100
        )
        roof_scale = (
            float(np.median(metric_scale[roof]))
            if np.any(roof)
            else 1.0
        )
        floor_scale = (
            float(np.median(metric_scale[floor]))
            if np.any(floor)
            else 1.0
        )
        median_anisotropy = (
            float(np.median(anisotropy))
            if len(anisotropy)
            else math.inf
        )
        anisotropy_p95 = (
            float(np.quantile(anisotropy, 0.95))
            if len(anisotropy)
            else math.inf
        )
        anisotropy_p99 = (
            float(np.quantile(anisotropy, 0.99))
            if len(anisotropy)
            else math.inf
        )
        severe_anisotropy_fraction = (
            float(np.mean(anisotropy > 25.0))
            if len(anisotropy)
            else 1.0
        )
        checks = [
            self._check(
                "uv",
                "Finite texture coordinates",
                bool(np.isfinite(texcoords).all()),
                f"vertices={len(texcoords)}",
            ),
            self._check(
                "uv",
                "Periodic seams use whole-tile offsets",
                integer_error <= 1e-4,
                f"maximum_integer_error={integer_error:.6g}",
            ),
            self._check(
                "uv",
                "Bounded UV chart seam overhead",
                (
                    len(faces) < 100
                    or chart_seam_vertices <= int(math.ceil(0.50 * len(positions)))
                ),
                (
                    f"groups={chart_seam_groups}, "
                    f"vertices={chart_seam_vertices}/{len(positions)}, "
                    f"fraction={chart_seam_vertices / max(len(positions), 1):.6g}"
                ),
            ),
            self._check(
                "uv",
                "Tangent continuity across wraps",
                maximum_tangent_angle <= 0.1,
                f"maximum_angle_degrees={maximum_tangent_angle:.6g}",
            ),
            self._check(
                "uv",
                "Pathological projection fraction",
                excessive_count <= excessive_limit,
                (
                    f"count={excessive_count}/{len(faces)}, "
                    f"fraction={excessive_fraction:.6g}, "
                    f"maximum_span={float(np.max(spans)):.6g}"
                ),
            ),
            self._check(
                "uv",
                "No collapsed UV triangles",
                collapsed_uv_count == 0,
                (
                    f"count={collapsed_uv_count}/{len(faces)}, "
                    f"minimum_abs_determinant={float(np.min(uv_determinants)):.6g}"
                ),
            ),
            self._check(
                "uv",
                "Metric roof/floor texel density",
                (
                    not enough_directional_faces
                    or (
                        0.70 <= roof_scale <= 1.30
                        and 0.70 <= floor_scale <= 1.30
                        and abs(roof_scale - floor_scale) <= 0.15
                    )
                ),
                (
                    f"roof_scale={roof_scale:.6g}, "
                    f"floor_scale={floor_scale:.6g}, "
                    f"target_scale_m={uv_scale_m:.6g}"
                ),
            ),
            self._check(
                "uv",
                "Localized UV anisotropy",
                (
                    len(anisotropy) < 100
                    or (
                        median_anisotropy <= 1.5
                        and anisotropy_p95 <= 5.0
                        and anisotropy_p99 <= 10.0
                        and severe_anisotropy_fraction <= 0.001
                    )
                ),
                (
                    f"median={median_anisotropy:.6g}, "
                    f"p95={anisotropy_p95:.6g}, "
                    f"p99={anisotropy_p99:.6g}, "
                    f"fraction_above_25={severe_anisotropy_fraction:.6g}, "
                    f"valid_faces={len(anisotropy)}/{len(faces)}"
                ),
            ),
        ]
        return checks

    def _scene_checks(self) -> list[ValidationCheck]:
        document = self.glb.document
        node_names = [str(node.get("name", "")) for node in document["nodes"]]
        event_nodes = [name for name in node_names if name.startswith("event_")]
        expected_events = self.manifest.get("events", ())
        expected_names = {
            str(event["node"])
            for event in expected_events
            if isinstance(event, dict) and "node" in event
        }
        cave_node, _mesh, _primitive = self.glb.cave_primitive()
        structural_expected = set(self.manifest.get("structural_event_ids", ()))
        structural_actual = set(
            cave_node.get("extras", {}).get("structural_event_ids", ())
        )
        return [
            self._check(
                "scene",
                "Complete event-node set",
                not expected_names or set(event_nodes) == expected_names,
                f"actual={len(event_nodes)}, expected={len(expected_names)}",
            ),
            self._check(
                "scene",
                "Unique scene node names",
                len(node_names) == len(set(node_names)),
                f"nodes={len(node_names)}",
            ),
            self._check(
                "scene",
                "Structural events recorded",
                structural_actual == structural_expected,
                f"actual={sorted(structural_actual)}, expected={sorted(structural_expected)}",
            ),
            self._check(
                "scene",
                "One mesh per editable scene object",
                len(document.get("meshes", ())) == 1 + len(event_nodes),
                (
                    f"meshes={len(document.get('meshes', ()))}, "
                    f"editable_objects={1 + len(event_nodes)}"
                ),
            ),
        ]

    def _displacement_checks(self) -> list[ValidationCheck]:
        cave_node, _mesh, primitive = self.glb.cave_primitive()
        displacement = cave_node.get("extras", {}).get("displacement", {})
        manifest_material = self.manifest.get("cave", {}).get("material", {})
        expected = bool(manifest_material.get("displacement_baked", False))
        baked = bool(displacement.get("baked", False))
        scale = float(displacement.get("scale_m", 0.0))
        minimum = float(displacement.get("minimum_offset_m", 0.0))
        maximum = float(displacement.get("maximum_offset_m", 0.0))
        deviation = float(displacement.get("sample_standard_deviation_m", 0.0))
        material = self.glb.document["materials"][primitive["material"]]
        minimum_width = float(
            self.manifest.get("summary", {}).get("minimum_section_width_m", 0.0)
        )
        clearance_fraction = (
            2.0 * scale / minimum_width
            if baked and minimum_width > 0.0
            else 0.0
        )
        return [
            self._check(
                "displacement",
                "Portable displacement contract",
                baked == expected,
                f"baked={baked}, expected={expected}",
            ),
            self._check(
                "displacement",
                "Displacement bounded in metres",
                not baked
                or (
                    scale > 0.0
                    and minimum >= -scale - 1e-6
                    and maximum <= scale + 1e-6
                ),
                f"range=[{minimum:.6g}, {maximum:.6g}], scale={scale:.6g}",
            ),
            self._check(
                "displacement",
                "Displacement is nonconstant",
                not baked or deviation > max(1e-5, scale * 0.01),
                f"sample_standard_deviation_m={deviation:.6g}",
            ),
            self._check(
                "displacement",
                "No nonstandard material displacement dependency",
                "displacementTexture" not in material,
                "height relief is stored in POSITION",
            ),
            self._check(
                "displacement",
                "Passage-clearance displacement budget",
                clearance_fraction <= 0.05,
                (
                    f"maximum_diameter_fraction={clearance_fraction:.6g}, "
                    f"minimum_section_width_m={minimum_width:.6g}"
                ),
            ),
        ]

    def _reproducibility_checks(self) -> list[ValidationCheck]:
        if not self.run_manifest:
            return [
                self._check(
                    "reproducibility",
                    "Run manifest available",
                    False,
                    f"missing={self.run_manifest_path}",
                )
            ]
        verified = 0
        records = self.run_manifest.get("outputs", ())
        for record in records:
            path = self.run_manifest_path.parent / record["path"]
            if path.is_file() and _sha256(path) == record["sha256"]:
                verified += 1
        return [
            self._check(
                "reproducibility",
                "Run completed",
                self.run_manifest.get("status") == "complete",
                f"status={self.run_manifest.get('status')}",
            ),
            self._check(
                "reproducibility",
                "Generated output hashes",
                bool(records) and verified == len(records),
                f"verified={verified}/{len(records)}",
            ),
        ]


def write_validation_reports(
    asset_path: str | Path,
    checks: list[ValidationCheck],
    output_directory: str | Path,
    *,
    pytest_exit_code: int | None = None,
) -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown summaries."""

    asset = Path(asset_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    valid = all(check.passed for check in checks) and pytest_exit_code in (None, 0)
    json_path = output / "validation_report.json"
    markdown_path = output / "validation_summary.md"
    payload = {
        "schema": "plume.portable-asset-validation.v1",
        "asset": str(asset),
        "valid": valid,
        "pytest_exit_code": pytest_exit_code,
        "passed": sum(check.passed for check in checks),
        "failed": sum(not check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# PLUME portable asset validation",
        "",
        f"- Asset: `{asset}`",
        f"- Result: **{'PASS' if valid else 'FAIL'}**",
        f"- Checks: {payload['passed']} passed, {payload['failed']} failed",
    ]
    if pytest_exit_code is not None:
        lines.append(f"- Pytest exit code: {pytest_exit_code}")
    lines.extend(("", "| Category | Check | Result | Detail |", "|---|---|---:|---|"))
    for check in checks:
        detail = check.detail.replace("|", "\\|")
        lines.append(
            f"| {check.category} | {check.name} | "
            f"{'PASS' if check.passed else 'FAIL'} | {detail} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


class _PytestProgressPlugin:
    def __init__(self, console: Console) -> None:
        self.progress = Progress(
            TextColumn("[bold cyan]Python regression tests"),
            BarColumn(bar_width=32),
            TaskProgressColumn(),
            TextColumn("({task.completed:.0f}/{task.total:.0f})"),
            TimeElapsedColumn(),
            console=console,
        )
        self.task: TaskID | None = None
        self.completed: set[str] = set()

    def pytest_collection_finish(self, session) -> None:
        self.progress.start()
        self.task = self.progress.add_task("tests", total=max(len(session.items), 1))

    def pytest_runtest_logreport(self, report) -> None:
        if self.task is None or report.nodeid in self.completed:
            return
        if report.when == "call" or (report.when in {"setup", "teardown"} and report.failed):
            self.completed.add(report.nodeid)
            self.progress.advance(self.task)

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if self.task is not None:
            self.progress.update(self.task, completed=self.progress.tasks[0].total)
        self.progress.stop()


def _run_pytest_with_progress(console: Console) -> int:
    try:
        import pytest
    except ImportError:
        console.print("[red]pytest is not installed; cannot run regression tests.[/red]")
        return 4
    plugin = _PytestProgressPlugin(console)
    return int(pytest.main(["-q"], plugins=[plugin]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_asset() -> Path:
    candidates = (
        Path("outputs/export_all/blender/plume_cave_scene.glb"),
        Path("outputs/export_neutral/plume_cave_scene.glb"),
        Path("outputs/export_blender/plume_cave_scene.glb"),
        Path("outputs/export_blender/stage_d_geometry.glb"),
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a portable PLUME GLB with visible progress.",
    )
    parser.add_argument("asset", type=Path, nargs="?", default=_default_asset())
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the Python regression suite with an individual-test progress bar.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    pytest_exit_code = (
        _run_pytest_with_progress(console)
        if args.run_tests
        else None
    )
    try:
        validator = PortableAssetValidator(
            args.asset,
            manifest_path=args.manifest,
            run_manifest_path=args.run_manifest,
        )
    except Exception as error:
        console.print(f"[red]Unable to parse {args.asset}: {type(error).__name__}: {error}[/red]")
        return 2

    progress = Progress(
        TextColumn("[bold cyan]Portable asset checks"),
        BarColumn(bar_width=32),
        TaskProgressColumn(),
        TextColumn("({task.completed:.0f}/{task.total:.0f})"),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[detail]}"),
        console=console,
    )
    with progress:
        task = progress.add_task(
            "validation",
            total=len(validator.PHASES),
            detail="starting",
        )

        def update(current: int, total: int, detail: str) -> None:
            progress.update(task, completed=current, total=total, detail=detail)

        checks = validator.validate(progress=update)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else validator.run_manifest_path.parent / "validation"
    )
    json_path, markdown_path = write_validation_reports(
        args.asset,
        checks,
        output_dir,
        pytest_exit_code=pytest_exit_code,
    )
    failed = [check for check in checks if not check.passed]
    if pytest_exit_code not in (None, 0):
        failed.append(
            ValidationCheck(
                "tests",
                "Python regression suite",
                False,
                f"exit_code={pytest_exit_code}",
            )
        )
    console.print(
        f"[{'green' if not failed else 'red'}]"
        f"{'PASS' if not failed else 'FAIL'}: "
        f"{len(checks) - sum(not check.passed for check in checks)}/{len(checks)} "
        f"asset checks passed.[/]"
    )
    console.print(f"JSON report: {json_path}")
    console.print(f"Markdown summary: {markdown_path}")
    return 0 if not failed else 1


__all__ = [
    "GlbAsset",
    "PortableAssetValidator",
    "ValidationCheck",
    "main",
    "write_validation_reports",
]


if __name__ == "__main__":
    raise SystemExit(main())
