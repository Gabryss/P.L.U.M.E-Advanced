"""Portable, offline 3D inspection pages for saved Stage-B network artifacts."""

from __future__ import annotations

import argparse
import html
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ASSETS = Path(__file__).with_name("web_assets")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Network coordinates and measurements must be finite numbers")
    return value


def viewer_network(payload, *, label="Network", root_seed=None, flow_angle=0., quality=None):
    """Keep every sample and the actual layer elevations, without generation state."""
    if payload.get("schema") != "plume.cave-network.v1":
        raise ValueError("Expected a PLUME network.json artifact (plume.cave-network.v1)")
    layered = bool(payload.get("layers", {}).get("controls", {}).get("enabled"))
    segments, identifiers = [], set()
    for raw in payload.get("segments", []):
        identifier = raw["segment_id"]
        if type(identifier) is not int or identifier in identifiers:
            raise ValueError("Segment identifiers must be unique integers")
        identifiers.add(identifier)
        samples = raw["centerline"]
        xyz = raw.get("centerline_xyz_m")
        if layered and xyz is None:
            raise ValueError("Layered networks require centerline_xyz_m; substrate elevation is not passage elevation")
        if xyz is None:
            xyz = [[p["x"], p["y"], p["elevation"]] for p in samples]
        if len(xyz) != len(samples) or len(xyz) < 2:
            raise ValueError("Each passage needs matching coordinates and at least two samples")
        coordinates = []
        for point in xyz:
            if len(point) != 3:
                raise ValueError("Network coordinates must be XYZ triples")
            coordinates.append([_number(v) for v in point])
        widths = [_number(p["width"]) for p in samples]
        if min(widths) <= 0:
            raise ValueError("Passage widths must be positive")
        meta = raw.get("metadata", {})
        layer = meta.get("regional_start_layer", raw.get("z_level", 0)) if layered else 0
        end_layer = meta.get("regional_end_layer", layer)
        if any(type(v) is not int or v < 0 for v in (layer, end_layer)):
            raise ValueError("Layer identifiers must be nonnegative integers")
        if any(type(raw[key]) is not int for key in ("source_node_id", "target_node_id")):
            raise ValueError("Passage endpoints must be integer node identifiers")
        if _number(raw["physical_length_m"]) < 0:
            raise ValueError("Passage lengths cannot be negative")
        segments.append(dict(id=identifier, source=raw["source_node_id"], target=raw["target_node_id"],
                             xyz=coordinates, widths=widths, layer=layer, endLayer=end_layer,
                             role=str(meta.get("regional_route_type", raw.get("kind", "passage"))),
                             length=_number(raw["physical_length_m"])))
    if not segments:
        raise ValueError("Cannot display an empty network")
    nodes = []
    ids = set()
    for node in payload.get("nodes", []):
        if type(node["node_id"]) is not int or node["node_id"] in ids:
            raise ValueError("Node identifiers must be unique integers")
        ids.add(node["node_id"])
        nodes.append(dict(id=node["node_id"], kind=str(node["kind"])))
    if any(s["source"] not in ids or s["target"] not in ids for s in segments):
        raise ValueError("Every passage endpoint must reference an existing network node")
    return dict(label=str(label), seed=root_seed, flowAngle=_number(flow_angle), nodes=nodes,
                segments=segments, fingerprint=str(payload.get("semantic_sha256", "")),
                quality=quality, layers=sorted({v for s in segments for v in (s["layer"], s["endLayer"])}))


def network_files(sources):
    """Read selected artifacts/campaigns, never recursively publish a workspace."""
    result = []
    for source in map(Path, sources):
        if source.is_file():
            result.append(source)
        elif (source / "network.json").is_file():
            result.append(source / "network.json")
        elif (source / "campaign.json").is_file():
            for case in json.loads((source / "campaign.json").read_text())["cases"]:
                if case.get("accepted"):
                    path = (source / case["case"] / "network.json").resolve()
                    if not path.is_relative_to(source.resolve()):
                        raise ValueError("Campaign case paths must stay inside the selected directory")
                    result.append(path)
        else:
            raise ValueError(f"No network.json or campaign.json at {source}")
    result = list(dict.fromkeys(p.resolve() for p in result))
    if not result:
        raise ValueError("No accepted network artifacts in the selected sources")
    return result


def export_network_viewer(sources, output, *, title="PLUME · Network explorer", labels=None):
    networks = []
    for path in network_files(sources):
        config_path, quality_path = path.with_name("resolved_config.json"), path.with_name("quality.json")
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        quality = json.loads(quality_path.read_text()) if quality_path.is_file() else {}
        network = viewer_network(json.loads(path.read_text()), label=path.parent.name,
                                 root_seed=config.get("procedural_seed"),
                                 flow_angle=config.get("host_field", {}).get("flow_angle_degrees", 0.),
                                 quality=quality.get("accepted"))
        if network["seed"] is not None:
            network["label"] = f"Seed {network['seed']} · {len(network['layers'])} layer(s)"
        if labels and str(path) in labels:
            network["label"] = str(labels[str(path)])
        networks.append(network)
    data = json.dumps(dict(networks=networks), separators=(",", ":"), allow_nan=False)
    # Embedded JSON is data, even if a supplied label contains an HTML end tag.
    data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = (ASSETS / "network_viewer.html").read_text()
    page = page.replace("__TITLE__", html.escape(title)).replace("__STYLE__", (ASSETS / "network_viewer.css").read_text())
    page = page.replace("__APP__", (ASSETS / "network_viewer.js").read_text()).replace("__DATA__", data)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def page_handler(page):
    """Serve only the generated page; no listings, neighboring files or symlinks."""
    content = Path(page).read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path not in ("/", "/index.html", "/viewer.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", type=Path, required=True, help="Network files, run folders or campaign folders")
    parser.add_argument("--output", type=Path, default=Path("outputs/network-viewer/index.html"))
    parser.add_argument("--serve", action="store_true", help="Keep a small read-only page server running")
    parser.add_argument("--bind", default="127.0.0.1", help="Listening address; set a trusted LAN/VPN interface for remote access")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        path = export_network_viewer(args.source, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Offline 3D viewer: {path.resolve()}", flush=True)
    if args.serve:
        with ThreadingHTTPServer((args.bind, args.port), page_handler(path)) as server:
            print(f"Serving only the viewer at http://{args.bind}:{args.port}/ (Ctrl+C to stop)", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
