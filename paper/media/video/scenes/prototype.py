"""Artifact-backed standalone stage scenes and the combined prototype."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AnimationGroup,
    Arrow,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    LaggedStart,
    Line,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Scene,
    Succession,
    ValueTracker,
    VGroup,
    VMobject,
    Wait,
    always_redraw,
)

from plume_advanced.media.manim_data import (
    ManimHostData,
    ManimPrototypeData,
    ManimSection,
    ManimSegment,
    evenly_spaced_sections,
    load_manim_host_data,
    load_manim_prototype_data,
)
from plume_advanced.media.manim_typography import PresentationText as Text

BACKGROUND = "#08131d"
TEXT = "#e7edf3"
MUTED = "#91a4b7"
NETWORK = "#26c6da"
SECTIONS = "#a78bfa"
GEOMETRY = "#64748b"
HIGHLIGHT = "#f59e0b"
ANNOTATION_FONT = "DejaVu Sans"
SAMPLE_MARKER = "#bef264"
SEGMENT_TYPE_COLORS = {
    "backbone": "#22d3ee",
    "source_feeder": "#fb923c",
    "chamber_braid": "#a78bfa",
    "inner_bypass": "#4ade80",
    "island_bypass": "#60a5fa",
    "ladder": "#f472b6",
    "spur": "#facc15",
}


class PipelineOverview(Scene):
    """Introduce the four artifact-backed stages before the detailed assets."""

    def construct(self) -> None:
        host = load_manim_host_data(_artifact_path("PLUME_VIDEO_HOST_ARTIFACT"))
        data = _prototype_data()
        title = _title("From host fields to continuous cave geometry")
        subtitle = Text(
            "Six stages · explicit artifacts · one reproducible pipeline",
            color=MUTED,
            font_size=20,
            font=ANNOTATION_FONT,
        ).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.0)

        colors = ("#38bdf8", NETWORK, SECTIONS, "#94a3b8", "#fb923c", "#4ade80")
        specifications = (
            (
                "A",
                "Host field",
                (
                    "Generate terrain and geological fields.",
                    "They define where cave growth is favored.",
                ),
                "routing-cost substrate",
            ),
            (
                "B",
                "Semantic network",
                (
                    "Propagate typed segments through the substrate.",
                    "Lava age controls the downstream flow.",
                ),
                "semantic centerline network",
            ),
            (
                "C",
                "Adaptive sections",
                (
                    "Sample local shape along every branch.",
                    "Sampling increases near bends and junctions.",
                ),
                "closed cross-section profiles",
            ),
            (
                "D",
                "Density and geometry",
                (
                    "Interpolate profile SDFs into voxel density.",
                    "Extract the connected, welded cave surface.",
                ),
                "connected welded cave mesh",
            ),
            (
                "E",
                "Rock placement",
                (
                    "Sample usable contacts on the cave floor.",
                    "Ground rock and boulder prop meshes.",
                ),
                "editable rock and boulder meshes",
            ),
            (
                "F",
                "Surface preparation",
                (
                    "Smooth, unwrap, and package the visual mesh.",
                    "Export one portable scene to target tools.",
                ),
                "render-ready portable scene",
            ),
        )
        rail, pills = _overview_progress(specifications, colors)
        active = RoundedRectangle(
            width=pills[0].width + 0.14,
            height=pills[0].height + 0.12,
            corner_radius=0.12,
            color=HIGHLIGHT,
            stroke_width=2.2,
            fill_opacity=0.0,
        ).move_to(pills[0])
        self.play(FadeIn(rail), FadeIn(active), run_time=0.8)
        self.wait(1.0)

        visuals = _overview_visuals(host, data)
        current = None
        for index, ((letter, name, lines, output), color, visual) in enumerate(
            zip(specifications, colors, visuals, strict=True)
        ):
            shell = _overview_focus_shell(letter, name, lines, output, color)
            incoming = Group(shell, visual)
            animations = [active.animate.move_to(pills[index]), FadeIn(shell)]
            if current is not None:
                animations.insert(0, FadeOut(current))
            self.play(*animations, run_time=0.8)
            self.play(FadeIn(visual, scale=0.94), run_time=1.0)
            self.wait(3.4)
            current = incoming

        recap = _overview_recap(specifications, colors, visuals)
        takeaway = Text(
            "Same cave, progressively enriched · each arrow passes a saved artifact",
            color=TEXT,
            font_size=19,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.28)
        self.play(
            FadeOut(current),
            FadeOut(active),
            FadeOut(rail),
            LaggedStart(*[FadeIn(card, shift=UP * 0.12) for card in recap[0]], lag_ratio=0.14),
            FadeIn(recap[1]),
            FadeIn(takeaway),
            run_time=1.6,
        )
        self.wait(5.5)


class StageAHostField(Scene):
    """Reveal the interpretable host layers consumed by network routing."""

    def construct(self) -> None:
        host = load_manim_host_data(_artifact_path("PLUME_VIDEO_HOST_ARTIFACT"))
        title = _title("Stage A · host-conditioned substrate")
        subtitle = Text(
            "Terrain provides spatial context; geological fields then shape routing",
            color=MUTED,
            font_size=20,
            font=ANNOTATION_FONT,
        ).next_to(title, DOWN, buff=0.15)
        title.set_z_index(10)
        subtitle.set_z_index(10)
        self.add(title, subtitle)
        self.add_foreground_mobjects(title, subtitle)

        elevation = host.fields["elevation"]
        mesh_data = _prepare_elevation_mesh(elevation)
        yaw = ValueTracker(0.0)
        orbit = ValueTracker(0.0)
        tilt = ValueTracker(0.0)
        relief = ValueTracker(0.0)
        layout = ValueTracker(0.0)
        surface = always_redraw(
            lambda: _elevation_surface(
                mesh_data,
                yaw=yaw.get_value() + np.deg2rad(18.0) * np.sin(orbit.get_value()),
                tilt=tilt.get_value(),
                relief=relief.get_value(),
                layout=layout.get_value(),
            )
        )
        surface_marker = always_redraw(
            lambda: _elevation_sample_marker(
                mesh_data,
                yaw=yaw.get_value() + np.deg2rad(18.0) * np.sin(orbit.get_value()),
                tilt=tilt.get_value(),
                relief=relief.get_value(),
                layout=layout.get_value(),
            )
        )
        elevation_image = ImageMobject(_field_rgba(elevation, "terrain"))
        elevation_image.width = 6.20
        elevation_image.move_to((-2.0, -0.28, 0.0))
        elevation_border = Rectangle(
            width=elevation_image.width + 0.08,
            height=elevation_image.height + 0.08,
            color="#475569",
            stroke_width=1.1,
        ).move_to(elevation_image)
        elevation_map = Group(elevation_image, elevation_border)
        map_context = _map_context(host, elevation_image)
        map_marker = _map_sample_marker(elevation_image)
        wireframe = _map_wireframe(mesh_data, elevation_image)
        elevation_info = Group(
            Text("Elevation", color=TEXT, font_size=29, font=ANNOTATION_FONT),
            VGroup(
                Text(
                    "color encodes height above",
                    color=MUTED,
                    font_size=17,
                    font=ANNOTATION_FONT,
                ),
                Text(
                    "the terrain datum",
                    color=MUTED,
                    font_size=17,
                    font=ANNOTATION_FONT,
                ),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.04),
            Text(
                f"range  {float(np.min(elevation)):.0f} to {float(np.max(elevation)):.0f} m",
                color=TEXT,
                font_size=17,
                font=ANNOTATION_FONT,
            ),
            _field_colorbar(elevation, "terrain", "elevation (m)", width=3.25),
            Text(
                "marker identifies one shared XY location",
                color=SAMPLE_MARKER,
                font_size=15,
                font=ANNOTATION_FONT,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        elevation_info.to_edge(RIGHT, buff=0.42).shift(DOWN * 0.05)
        self.play(
            FadeIn(elevation_map),
            FadeIn(map_context),
            FadeIn(map_marker),
            FadeIn(elevation_info),
            run_time=1.4,
        )
        self.wait(5.2)

        transition_note = Text(
            "First expose the sampling grid; then lift every sample by elevation",
            color=TEXT,
            font_size=20,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.30)
        self.play(FadeIn(wireframe), FadeIn(transition_note), run_time=1.2)
        self.add_foreground_mobject(transition_note)
        self.wait(3.0)
        self.play(
            FadeOut(elevation_map),
            FadeIn(surface),
            FadeOut(map_context),
            FadeOut(map_marker),
            FadeOut(elevation_info),
            run_time=1.0,
        )
        self.play(
            tilt.animate.set_value(np.deg2rad(58.0)),
            relief.animate.set_value(1.0),
            yaw.animate.set_value(np.deg2rad(28.0)),
            layout.animate.set_value(0.45),
            FadeOut(wireframe),
            run_time=5.0,
        )
        vertical_scale = Text(
            "vertical relief exaggerated for readability",
            color=MUTED,
            font_size=16,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.34)
        self.play(FadeOut(transition_note), FadeIn(vertical_scale), run_time=0.7)
        self.wait(3.0)
        self.add(orbit)
        orbit.add_updater(lambda tracker, dt: tracker.increment_value(0.45 * dt))

        input_specifications = (
            (
                "cover_thickness",
                "Cover thickness",
                "available roof cover above the tube",
                "cover",
                "metres",
            ),
            (
                "fracture_intensity",
                "Fracture intensity",
                "structural weakness and preferred corridors",
                "fracture",
                "0–1 index",
            ),
            (
                "roof_stability",
                "Roof stability",
                "gravity- and material-aware stability proxy",
                "stability",
                "0–1 index",
            ),
        )
        focus_cards = [
            _host_field_focus(host, field, label, description, palette, unit)
            for field, label, description, palette, unit in input_specifications
        ]
        relief_label = Text(
            "bounded terrain orbit · same-location marker",
            color=MUTED,
            font_size=16,
            font=ANNOTATION_FONT,
        ).move_to((-3.55, 1.75, 0.0))
        self.play(
            FadeOut(vertical_scale),
            FadeIn(relief_label),
            FadeIn(surface_marker),
            layout.animate.set_value(1.0),
            run_time=1.5,
        )
        current = focus_cards[0]
        self.play(FadeIn(current), run_time=0.9)
        self.wait(4.5)
        for next_card in focus_cards[1:]:
            self.play(FadeOut(current), FadeIn(next_card), run_time=0.9)
            current = next_card
            self.wait(4.5)

        derivation_parts = _routing_derivation(host)
        derivation = Group(*derivation_parts)
        derivation_heading, derivation_terms, derivation_combine, derivation_result, derivation_scale = (
            derivation_parts
        )
        self.play(
            FadeOut(current),
            FadeIn(derivation_heading),
            FadeIn(derivation_terms),
            run_time=1.2,
        )
        self.wait(2.0)
        self.play(FadeIn(derivation_combine), run_time=0.8)
        self.wait(1.4)
        self.play(FadeIn(derivation_result), FadeIn(derivation_scale), run_time=1.2)
        self.wait(4.2)

        specifications = (*input_specifications, (
            "routing_cost",
            "Routing cost",
            "derived weighted penalty",
            "cost",
            "0–1 cost",
        ))
        grid = _host_field_grid(host, specifications)
        takeaway = Text(
            "Shared marker across every map · saved routing cost passes to Stage B",
            color=TEXT,
            font_size=19,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.28)
        self.play(FadeOut(derivation), FadeIn(grid), FadeIn(takeaway), run_time=1.3)
        self.wait(7.0)
        orbit.clear_updaters()


class StageBSemanticFlow(Scene):
    """Reveal the semantic network in propagated lava-flow order."""

    def construct(self) -> None:
        host = load_manim_host_data(_artifact_path("PLUME_VIDEO_HOST_ARTIFACT"))
        data = _prototype_data()
        projector = PlanProjector(tuple(data.segments), rotation_radians=-0.5 * np.pi)
        title = _title("Stage B · semantic cave network")
        handoff_subtitle = Text(
            "Stage A supplies the host-conditioned routing substrate",
            color=MUTED,
            font_size=20,
            font=ANNOTATION_FONT,
        ).next_to(title, DOWN, buff=0.15)
        flow_subtitle = Text(
            "Propagation follows lava age; color denotes each segment's semantic role",
            color=MUTED,
            font_size=20,
            font=ANNOTATION_FONT,
        ).next_to(title, DOWN, buff=0.15)
        self.add(title, handoff_subtitle)

        host_handoff = _stage_b_host_handoff(host)
        handoff_note = Text(
            "The network inherits this terrain- and geology-aware cost field.",
            color=TEXT,
            font_size=18,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.30)
        self.play(FadeIn(host_handoff), FadeIn(handoff_note), run_time=1.2)
        self.wait(3.8)
        self.play(
            FadeOut(host_handoff),
            FadeOut(handoff_note),
            FadeOut(handoff_subtitle),
            FadeIn(flow_subtitle),
            run_time=1.0,
        )

        substrate = _routing_cost_overlay(host, projector)
        ghost = VGroup(
            *[_segment_path(segment, projector, "#334155", 1.5) for segment in data.segments]
        ).set_stroke(opacity=0.58).set_z_index(1)
        self.play(FadeIn(substrate), FadeIn(ghost), run_time=1.0)
        flow_paths = VGroup(
            *[
                _segment_path(segment, projector, _segment_type_color(segment.kind), 3.2)
                for segment in data.segments
            ]
        ).set_z_index(2)
        type_legend = _segment_type_legend(data.segments)
        self.play(FadeIn(type_legend), run_time=0.5)
        scheduled = _flow_animations(data.segments, tuple(flow_paths), total_time=8.0)
        self.play(AnimationGroup(*scheduled, lag_ratio=0.0))
        flow_note = Text(
            "Reveal order: source feeders → splits → braids → merges → downstream route",
            color=TEXT,
            font_size=18,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.30)
        self.play(FadeIn(flow_note), run_time=0.8)
        self.wait(2.0)


class StageCAdaptiveSections(Scene):
    """Give the adaptive-section explanation its own slower reading sequence."""

    def construct(self) -> None:
        data = _prototype_data()
        projector = PlanProjector(tuple(data.segments), rotation_radians=-0.5 * np.pi)
        section_segment_id = data.longest_section_segment_id
        section_segment = next(
            segment for segment in data.segments if segment.segment_id == section_segment_id
        )
        section_samples = tuple(
            section for section in data.sections if section.segment_id == section_segment_id
        )
        title = _title("Stage C · adaptive cross-sections")
        subtitle = Text(
            "One long branch, sampled from upstream to downstream",
            color=MUTED,
            font_size=20,
            font=ANNOTATION_FONT,
        ).next_to(title, DOWN, buff=0.15)
        self.add(title, subtitle)

        network = VGroup(
            *[_segment_path(segment, projector, NETWORK, 1.8) for segment in data.segments]
        ).set_stroke(opacity=0.56)
        hero = _segment_path(section_segment, projector, HIGHLIGHT, 5.0)
        explanation = VGroup(
            Text("Each profile controls:", color=TEXT, font_size=20, font=ANNOTATION_FONT),
            Text(
                "width · height · roof arch · floor relief",
                color=SECTIONS,
                font_size=19,
                font=ANNOTATION_FONT,
            ),
            Text(
                "Samples become denser near curvature and junctions.",
                color=MUTED,
                font_size=17,
                font=ANNOTATION_FONT,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        explanation.to_corner(UP + RIGHT, buff=0.48).shift(DOWN * 0.75)
        self.play(FadeIn(network), run_time=1.0)
        self.play(Create(hero), FadeIn(explanation), run_time=1.5)
        self.wait(2.4)

        selected = evenly_spaced_sections(section_samples, 7)
        panel, profiles = _section_panel(
            selected,
            width=7.8,
            height=1.9,
            caption="Same branch · morphology sampled along distance",
        )
        tracker = Dot(hero.get_start(), radius=0.085, color=SAMPLE_MARKER)
        tracker.set_stroke(color=TEXT, width=1.5, opacity=0.95)
        self.play(FadeIn(panel), FadeIn(tracker), run_time=0.8)
        current_connector = None
        current_progress = None
        for index, profile in enumerate(profiles):
            alpha = index / max(len(profiles) - 1, 1)
            connector = always_redraw(
                lambda profile=profile: DashedLine(
                    tracker.get_center(),
                    profile.get_top() + UP * 0.04,
                    color=SAMPLE_MARKER,
                    stroke_width=1.5,
                    stroke_opacity=0.72,
                    dash_length=0.10,
                    dashed_ratio=0.55,
                ).set_z_index(2)
            )
            if current_connector is not None:
                self.remove(current_connector)
            self.add(connector)
            progress = _section_progress_badge(alpha)
            progress.next_to(panel, UP, buff=0.10).align_to(panel, LEFT)
            animations = [
                tracker.animate.move_to(hero.point_from_proportion(alpha)),
                Create(profile),
                FadeIn(progress),
            ]
            if current_progress is not None:
                animations.append(FadeOut(current_progress))
            self.play(*animations, run_time=1.0)
            current_connector = connector
            current_progress = progress
        reading_note = Text(
            "Closed local profiles become the input to Stage D volume construction.",
            color=TEXT,
            font_size=18,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.18).to_edge(LEFT, buff=0.35)
        self.play(FadeIn(reading_note), run_time=0.6)
        self.wait(3.2)


class StageDMarchingCubes(Scene):
    """Explain PLUME's chunked Lewiner marching-cubes meshing pipeline."""

    def construct(self) -> None:
        data = _prototype_data()
        section_segment_id = data.longest_section_segment_id
        section_samples = tuple(
            section for section in data.sections if section.segment_id == section_segment_id
        )
        selected = evenly_spaced_sections(section_samples, 6)
        title = _title("Stage D · density field to cave mesh")
        subtitle = Text(
            "Chunked Lewiner marching cubes extracts the zero-level isosurface",
            color=MUTED,
            font_size=20,
            font=ANNOTATION_FONT,
        ).next_to(title, DOWN, buff=0.15)
        self.add(title, subtitle)
        progress, progress_nodes = _stage_d_progress()
        progress_active = RoundedRectangle(
            width=progress_nodes[0].width + 0.10,
            height=progress_nodes[0].height + 0.08,
            corner_radius=0.10,
            color=HIGHLIGHT,
            stroke_width=1.8,
            fill_opacity=0.0,
        ).move_to(progress_nodes[0])
        self.play(FadeIn(progress), FadeIn(progress_active), run_time=0.7)

        step_one = _stage_d_step(
            "1 · interpolate neighboring profile SDFs into voxel density"
        )
        density_slice = _profile_density_slice(selected[len(selected) // 2])
        density_key = _density_key()
        self.play(FadeIn(step_one), run_time=0.5)
        self.play(
            progress_active.animate.move_to(progress_nodes[1]),
            LaggedStart(*[FadeIn(dot) for dot in density_slice[0]], lag_ratio=0.008),
            Create(density_slice[1]),
            FadeIn(density_key),
            run_time=2.0,
        )
        self.wait(1.2)

        step_two = _stage_d_step("2 · classify each cube corner against iso-level 0")
        (
            cube_faces,
            cube_edges,
            corner_dots,
            corner_labels,
            sign_edges,
            crossings,
            triangles,
        ) = _marching_cube_case()
        self.play(
            progress_active.animate.move_to(progress_nodes[2]),
            FadeOut(step_one),
            FadeOut(density_slice),
            FadeOut(density_key),
            FadeIn(step_two),
            run_time=0.7,
        )
        depth_key = Text(
            "solid edges = front   ·   dashed edges = rear",
            color=MUTED,
            font_size=15,
            font=ANNOTATION_FONT,
        ).to_corner(DOWN + LEFT, buff=0.42)
        self.play(FadeIn(cube_faces), Create(cube_edges), FadeIn(depth_key), run_time=1.0)
        self.play(
            LaggedStart(*[FadeIn(dot) for dot in corner_dots], lag_ratio=0.10),
            FadeIn(corner_labels),
            run_time=1.4,
        )
        sign_key = _stage_d_side_note(
            ("positive density = carved void", "negative density = rock"), MUTED
        )
        self.play(FadeIn(sign_key), run_time=0.5)
        self.wait(1.5)
        mixed_note = _stage_d_side_note(
            ("Only opposite-sign edges", "can cross the d = 0 surface"), HIGHLIGHT
        )
        self.play(
            FadeOut(sign_key),
            FadeIn(mixed_note),
            LaggedStart(*[Create(edge) for edge in sign_edges], lag_ratio=0.15),
            run_time=1.4,
        )
        self.wait(1.6)

        step_three = _stage_d_step(
            "3a · interpolate d = 0 on every sign-changing edge"
        )
        self.play(FadeOut(step_two), FadeIn(step_three), run_time=0.5)
        interpolation = VGroup(
            Text("edge interpolation", color=TEXT, font_size=17, font=ANNOTATION_FONT),
            Text("t = (0 − d₀) / (d₁ − d₀)", color=HIGHLIGHT, font_size=19, font=ANNOTATION_FONT),
            Text("vertex = p₀ + t (p₁ − p₀)", color=MUTED, font_size=16, font=ANNOTATION_FONT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        interpolation.to_edge(RIGHT, buff=0.42).shift(UP * 0.15)
        self.play(FadeIn(interpolation), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(dot) for dot in crossings], lag_ratio=0.22),
            run_time=1.8,
        )
        self.wait(1.5)

        triangulation_step = _stage_d_step(
            "3b · connect crossings using the Lewiner case topology"
        )
        patch_note = _stage_d_side_note(
            ("Two triangles form", "this cube's surface patch"), NETWORK
        )
        self.play(
            progress_active.animate.move_to(progress_nodes[3]),
            FadeOut(step_three),
            FadeIn(triangulation_step),
            FadeOut(interpolation),
            FadeOut(mixed_note),
            run_time=0.6,
        )
        self.play(FadeIn(triangles[0]), run_time=1.1)
        self.play(FadeIn(triangles[1]), FadeIn(patch_note), run_time=1.1)
        self.wait(1.8)

        step_four = _stage_d_step("4 · assemble chunks and weld coincident boundary vertices")
        tube_faces, tube_wire, seam = _section_tube_mesh(selected)
        cube_case = VGroup(
            cube_faces,
            cube_edges,
            corner_dots,
            corner_labels,
            sign_edges,
            crossings,
            triangles,
            depth_key,
        )
        self.play(
            progress_active.animate.move_to(progress_nodes[4]),
            FadeOut(triangulation_step),
            FadeOut(patch_note),
            FadeOut(cube_case),
            FadeIn(step_four),
            run_time=0.7,
        )
        self.play(LaggedStart(*[FadeIn(face) for face in tube_faces], lag_ratio=0.012), run_time=2.0)
        self.play(
            LaggedStart(*[Create(line) for line in tube_wire], lag_ratio=0.015),
            FadeIn(seam),
            run_time=1.3,
        )
        weld_note = Text(
            "duplicate seam vertices",
            color="#f472b6",
            font_size=16,
            font=ANNOTATION_FONT,
        ).next_to(seam, UP, buff=0.12)
        self.play(FadeIn(weld_note), run_time=0.4)
        self.play(FadeOut(seam), FadeOut(weld_note), tube_wire.animate.set_color(NETWORK), run_time=1.0)
        note = Text(
            "Result: one connected, welded triangle surface in world coordinates",
            color=TEXT,
            font_size=18,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.28)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(2.4)


class StageEGeologicalEvents(Scene):
    """Explain how rock and boulder prop meshes are grounded on the cave floor."""

    def construct(self) -> None:
        data = _prototype_data()
        section_id = data.longest_section_segment_id
        sections = tuple(
            section for section in data.sections if section.segment_id == section_id
        )
        selected = evenly_spaced_sections(sections, 6)
        representative = selected[len(selected) // 2]
        title = _title("Stage E · grounded cave rocks")
        subtitle = Text(
            "Surface-aware placement grounds separately editable rocks and boulders",
            color=MUTED,
            font_size=20,
            font=ANNOTATION_FONT,
        ).next_to(title, DOWN, buff=0.15)
        self.add(title, subtitle)

        flow, flow_nodes = _event_interleave_flow()
        active = RoundedRectangle(
            width=flow_nodes[0].width + 0.13,
            height=flow_nodes[0].height + 0.11,
            corner_radius=0.11,
            color=HIGHLIGHT,
            stroke_width=2.0,
            fill_opacity=0.0,
        ).move_to(flow_nodes[0])
        self.play(FadeIn(flow), FadeIn(active), run_time=0.8)

        before, _before_props = _event_cross_section_parts(
            representative,
            center_x=-3.25,
            maximum_width=4.25,
        )
        after, props = _event_cross_section_parts(
            representative,
            center_x=3.25,
            maximum_width=4.25,
        )
        before_label = VGroup(
            Text("BEFORE", color=MUTED, font_size=14, font=ANNOTATION_FONT),
            Text("Stage D cave surface", color=TEXT, font_size=21, font=ANNOTATION_FONT),
        ).arrange(DOWN, buff=0.05)
        before_label.move_to((-3.25, 1.13, 0.0))
        after_label = VGroup(
            Text("AFTER", color="#fb923c", font_size=14, font=ANNOTATION_FONT),
            Text("grounded rock props", color=TEXT, font_size=21, font=ANNOTATION_FONT),
        ).arrange(DOWN, buff=0.05)
        after_label.move_to((3.25, 1.13, 0.0))
        comparison_arrow = VGroup(
            Text("→", color=HIGHLIGHT, font_size=31, font=ANNOTATION_FONT),
            Text("ground", color=MUTED, font_size=13, font=ANNOTATION_FONT),
        ).arrange(DOWN, buff=0.01).move_to((0.0, -0.30, 0.0))
        self.play(FadeIn(before), FadeIn(before_label), run_time=1.0)
        self.wait(2.4)
        self.play(
            active.animate.move_to(flow_nodes[1]),
            FadeIn(after),
            FadeIn(after_label),
            FadeIn(comparison_arrow),
            run_time=0.8,
        )
        for dot, target in props:
            self.add(dot)
            self.play(dot.animate.move_to(target), run_time=0.7)
        grounding_note = Text(
            "Floor contact provides position and normal; each prop is aligned and slightly embedded.",
            color=MUTED,
            font_size=17,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.62)
        self.play(FadeIn(grounding_note), run_time=0.7)
        self.wait(2.8)
        self.play(
            active.animate.move_to(flow_nodes[2]),
            run_time=0.8,
        )
        takeaway = Text(
            "Cave topology is preserved; rock and boulder meshes remain separately editable.",
            color=TEXT,
            font_size=18,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(grounding_note), FadeIn(takeaway), run_time=0.7)
        self.wait(3.5)


class StageFSurfacePreparation(Scene):
    """Explain the implemented visual-surface and portable export path."""

    def construct(self) -> None:
        data = _prototype_data()
        section_id = data.longest_section_segment_id
        section_samples = tuple(
            section for section in data.sections if section.segment_id == section_id
        )
        selected = evenly_spaced_sections(section_samples, 7)
        title = _title("Stage F · surface preparation and export")
        subtitle = Text(
            "The welded topology becomes a portable, render-ready visual asset",
            color=MUTED,
            font_size=20,
            font=ANNOTATION_FONT,
        ).next_to(title, DOWN, buff=0.15)
        self.add(title, subtitle)

        cards, arrows, targets = _surface_preparation_parts(compact=False)
        scope = Text(
            "Implemented path: smoothing · displacement baking · UVs · tangents · embedded PBR maps",
            color=MUTED,
            font_size=16,
            font=ANNOTATION_FONT,
        ).move_to((0.0, 2.05, 0.0))
        self.play(FadeIn(scope), run_time=0.7)
        self.play(FadeIn(cards[0], shift=RIGHT * 0.15), run_time=1.0)
        self.wait(2.5)
        for index in range(1, len(cards)):
            self.play(
                FadeIn(arrows[index - 1]),
                FadeIn(cards[index], shift=RIGHT * 0.15),
                run_time=1.0,
            )
            self.wait(2.7)

        self.play(LaggedStart(*[FadeIn(target) for target in targets], lag_ratio=0.10), run_time=1.0)
        self.wait(2.4)
        boundary = Text(
            "Not claimed here: geology-conditioned synthesis or explicit visual LOD generation",
            color="#fbbf24",
            font_size=16,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.28)
        self.play(FadeIn(boundary), run_time=0.7)
        self.wait(4.0)

        asset_subtitle = Text(
            "Profile-backed schematic · rotating geometry with illustrative surface styles",
            color=MUTED,
            font_size=20,
            font=ANNOTATION_FONT,
        ).next_to(title, DOWN, buff=0.15)
        self.play(
            FadeOut(Group(*cards, arrows, targets, scope, boundary)),
            FadeOut(subtitle),
            FadeIn(asset_subtitle),
            run_time=1.0,
        )
        channel_names = ("geometry", "displacement", "normals + PBR", "packaged scene")
        channel_row, channel_nodes = _surface_channel_progress(channel_names)
        channel_active = RoundedRectangle(
            width=channel_nodes[0].width + 0.10,
            height=channel_nodes[0].height + 0.08,
            corner_radius=0.10,
            color=HIGHLIGHT,
            stroke_width=1.8,
            fill_opacity=0.0,
        ).move_to(channel_nodes[0])
        orbit = ValueTracker(-0.55)
        view = always_redraw(
            lambda: _prepared_surface_view(selected, orbit.get_value(), channel_names[0])
        )
        asset_note = Text(
            "Schematic styles, not exported texture channels · geometry from frozen Stage-C profiles",
            color=MUTED,
            font_size=15,
            font=ANNOTATION_FONT,
        ).to_edge(DOWN, buff=0.28)
        self.add(view)
        self.play(FadeIn(channel_row), FadeIn(channel_active), FadeIn(asset_note), run_time=0.8)
        self.add(orbit)
        orbit.add_updater(lambda tracker, dt: tracker.increment_value(0.12 * dt))
        self.wait(2.5)
        for index, channel_name in enumerate(channel_names[1:], start=1):
            self.remove(view)
            view = always_redraw(
                lambda channel_name=channel_name: _prepared_surface_view(
                    selected,
                    orbit.get_value(),
                    channel_name,
                )
            )
            self.add(view)
            self.play(channel_active.animate.move_to(channel_nodes[index]), run_time=0.65)
            self.wait(2.5)
        orbit.clear_updaters()


class GraphToGeometryPrototype(Scene):
    """Track one physical branch from graph through sections to carved volume."""

    def construct(self) -> None:
        data = load_manim_prototype_data(
            _artifact_path("PLUME_VIDEO_NETWORK_ARTIFACT"),
            _artifact_path("PLUME_VIDEO_SECTION_ARTIFACT"),
            hero_segment_id=_optional_int("PLUME_VIDEO_HERO_SEGMENT"),
        )
        projector = PlanProjector(tuple(data.segments))
        title = Text("From topology to continuous cave geometry", color=TEXT, font_size=36)
        title.to_edge(UP, buff=0.30)
        stage = Text("Stage B · semantic cave network", color=NETWORK, font_size=25)
        stage.next_to(title, DOWN, buff=0.15)
        self.add(title, stage)

        network_lines = VGroup(
            *[_segment_path(segment, projector, NETWORK, 2.2) for segment in data.segments]
        )
        network_lines.set_z_index(2)
        hero_line = _segment_path(data.hero_segment, projector, HIGHLIGHT, 5.0)
        hero_line.set_z_index(3)
        self.play(
            LaggedStart(*[Create(line) for line in network_lines], lag_ratio=0.035),
            run_time=3.0,
        )
        self.play(Create(hero_line), run_time=0.8)

        sections_label = Text(
            "Stage C · adaptive cross-sections",
            color=SECTIONS,
            font_size=25,
        ).move_to(stage)
        panel, profiles = _section_panel(
            evenly_spaced_sections(data.hero_sections, 5),
        )
        tracker = Dot(hero_line.get_start(), radius=0.065, color=HIGHLIGHT)
        self.play(FadeOut(stage), FadeIn(sections_label), FadeIn(panel), run_time=0.6)
        self.add(tracker)
        for index, profile in enumerate(profiles):
            alpha = index / max(len(profiles) - 1, 1)
            self.play(
                tracker.animate.move_to(hero_line.point_from_proportion(alpha)),
                Create(profile),
                run_time=0.7,
            )
        self.wait(0.4)

        geometry_label = Text(
            "Stage D · section-derived cave envelope",
            color=TEXT,
            font_size=25,
        ).move_to(stage)
        passage_envelopes = _passage_envelopes(data, projector)
        passage_envelopes.set_z_index(0)
        self.play(
            FadeOut(sections_label),
            FadeIn(geometry_label),
            FadeOut(panel),
            FadeOut(VGroup(*profiles)),
            FadeOut(tracker),
            FadeIn(passage_envelopes),
            network_lines.animate.set_stroke(opacity=0.68),
            run_time=1.4,
        )
        self.play(hero_line.animate.set_stroke(width=2.5), run_time=0.4)

        provenance = Text(
            (
                f"network {data.network_semantic_sha256[:10]}  ·  "
                f"sections {data.section_semantic_sha256[:10]}"
            ),
            color=MUTED,
            font_size=15,
            font=ANNOTATION_FONT,
        )
        provenance.to_edge(DOWN, buff=0.20).to_edge(RIGHT, buff=0.35)
        self.play(FadeIn(provenance), run_time=0.4)
        self.wait(1.5)


def _overview_progress(
    specifications: tuple[tuple[str, str, tuple[str, str], str], ...],
    colors: tuple[str, ...],
) -> tuple[VGroup, tuple[VGroup, ...]]:
    """Build the persistent A-to-D progress rail."""

    pills = []
    compact = len(specifications) > 4
    pill_width = 1.72 if compact else 2.45
    short_names = {
        "Host field": "Host",
        "Semantic network": "Network",
        "Adaptive sections": "Sections",
        "Density and geometry": "Geometry",
        "Rock placement": "Rocks",
        "Surface preparation": "Surface",
    }
    for (letter, name, _lines, _output), color in zip(
        specifications, colors, strict=True
    ):
        box = RoundedRectangle(
            width=pill_width,
            height=0.54,
            corner_radius=0.10,
            color="#334155",
            fill_color="#10202e",
            fill_opacity=0.96,
            stroke_width=1.0,
        )
        label = Text(
            f"{letter} · {short_names.get(name, name)}",
            color=color,
            font_size=13 if compact else 15,
            font=ANNOTATION_FONT,
        )
        pills.append(VGroup(box, label))
    pill_row = VGroup(*pills).arrange(RIGHT, buff=0.40 if compact else 0.58)
    pill_row.move_to((0.0, 2.02, 0.0))
    arrows = VGroup(
        *[
            Arrow(
                pills[index].get_right() + RIGHT * 0.06,
                pills[index + 1].get_left() + LEFT * 0.06,
                color=MUTED,
                stroke_width=1.4,
                max_tip_length_to_length_ratio=0.30,
            )
            for index in range(len(pills) - 1)
        ]
    )
    return VGroup(arrows, pill_row), tuple(pills)


def _overview_focus_shell(
    letter: str,
    name: str,
    lines: tuple[str, str],
    output: str,
    color: str,
) -> VGroup:
    """Create the common explanatory frame for one overview stage."""

    frame = RoundedRectangle(
        width=12.55,
        height=4.15,
        corner_radius=0.18,
        color="#334155",
        fill_color="#0b1925",
        fill_opacity=0.88,
        stroke_width=1.1,
    ).move_to((0.0, -0.48, 0.0))
    divider = Line(
        np.asarray((0.0, 1.20, 0.0)),
        np.asarray((0.0, -2.15, 0.0)),
        color="#334155",
        stroke_width=1.0,
    )
    heading = VGroup(
        Text(f"STAGE {letter}", color=color, font_size=17, font=ANNOTATION_FONT),
        Text(name, color=TEXT, font_size=30, font=ANNOTATION_FONT),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
    description = VGroup(
        *[
            Text(line, color=MUTED, font_size=18, font=ANNOTATION_FONT)
            for line in lines
        ]
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.09)
    output_group = VGroup(
        Text("OUTPUT", color=color, font_size=13, font=ANNOTATION_FONT),
        Text(output, color=TEXT, font_size=19, font=ANNOTATION_FONT),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
    text = VGroup(heading, description, output_group).arrange(
        DOWN, aligned_edge=LEFT, buff=0.30
    )
    text.move_to((3.25, -0.45, 0.0))
    return VGroup(frame, divider, text)


def _overview_visuals(
    host: ManimHostData,
    data: ManimPrototypeData,
) -> tuple[Group | VGroup, ...]:
    """Build compact visuals from the same artifacts as the detailed stages."""

    elevation = ImageMobject(_field_rgba(host.fields["elevation"], "terrain"))
    elevation.width = 2.20
    cost = ImageMobject(_field_rgba(host.fields["routing_cost"], "cost"))
    cost.width = 2.20
    elevation_card = Group(
        Text("host fields", color="#38bdf8", font_size=16, font=ANNOTATION_FONT),
        Group(
            elevation,
            Rectangle(
                width=elevation.width + 0.05,
                height=elevation.height + 0.05,
                color="#475569",
                stroke_width=0.9,
            ).move_to(elevation),
        ),
    ).arrange(DOWN, buff=0.10)
    cost_card = Group(
        Text("routing cost", color=HIGHLIGHT, font_size=16, font=ANNOTATION_FONT),
        Group(
            cost,
            Rectangle(
                width=cost.width + 0.05,
                height=cost.height + 0.05,
                color=HIGHLIGHT,
                stroke_width=1.1,
            ).move_to(cost),
        ),
    ).arrange(DOWN, buff=0.10)
    stage_a = Group(
        elevation_card,
        Text("→", color=MUTED, font_size=26, font=ANNOTATION_FONT),
        cost_card,
    ).arrange(RIGHT, buff=0.18)
    stage_a.move_to((-3.25, -0.48, 0.0))

    projector = PlanProjector(tuple(data.segments), rotation_radians=-0.5 * np.pi)
    stage_b = VGroup(
        *[
            _segment_path(segment, projector, _segment_type_color(segment.kind), 3.0)
            for segment in data.segments
        ]
    )
    stage_b.set_width(4.65)
    if stage_b.height > 3.10:
        stage_b.set_height(3.10)
    stage_b.move_to((-3.25, -0.48, 0.0))

    section_id = data.longest_section_segment_id
    sections = tuple(
        section for section in data.sections if section.segment_id == section_id
    )
    selected = evenly_spaced_sections(sections, 6)
    section_panel, profiles = _section_panel(
        selected,
        width=5.2,
        height=2.1,
        caption="adaptive profiles along one branch",
    )
    stage_c = Group(section_panel, *profiles)
    stage_c.set_width(4.85)
    stage_c.move_to((-3.25, -0.48, 0.0))

    cube_faces, cube_edges, _corner_dots, _corner_labels, _sign_edges, crossings, triangles = (
        _marching_cube_case()
    )
    stage_d = VGroup(cube_faces, cube_edges, crossings, triangles)
    stage_d.set_height(3.05)
    stage_d.move_to((-3.25, -0.48, 0.0))
    stage_e = _geological_event_cross_section(selected[len(selected) // 2], compact=True)
    stage_e.move_to((-3.25, -0.48, 0.0))
    stage_f = _surface_preparation_strip(compact=True)
    stage_f.move_to((-3.25, -0.48, 0.0))
    return stage_a, stage_b, stage_c, stage_d, stage_e, stage_f


def _overview_recap(
    specifications: tuple[tuple[str, str, tuple[str, str], str], ...],
    colors: tuple[str, ...],
    visuals: tuple[Group | VGroup, ...],
) -> tuple[Group, VGroup]:
    """Create a same-cave recap with one artifact thumbnail per stage."""

    cards = []
    compact = len(specifications) > 4
    card_width = 1.95 if compact else 2.82
    short_outputs = {
        "routing-cost substrate": "routing cost",
        "semantic centerline network": "network",
        "closed cross-section profiles": "profiles",
        "connected welded cave mesh": "cave mesh",
        "editable rock and boulder meshes": "rock props",
        "render-ready portable scene": "portable scene",
    }
    for (letter, name, _lines, output), color, source_visual in zip(
        specifications, colors, visuals, strict=True
    ):
        box = RoundedRectangle(
            width=card_width,
            height=3.05,
            corner_radius=0.16,
            color=color,
            fill_color="#0d1c29",
            fill_opacity=0.94,
            stroke_width=1.4,
        )
        stage = Text(letter, color=color, font_size=36, font=ANNOTATION_FONT)
        display_name = {
            "Semantic network": "Network",
            "Adaptive sections": "Sections",
            "Density and geometry": "Geometry",
            "Rock placement": "Rocks",
            "Surface preparation": "Surface",
        }.get(name, name)
        title = Text(
            display_name,
            color=TEXT,
            font_size=16 if compact else 19,
            font=ANNOTATION_FONT,
        )
        thumbnail = source_visual.copy()
        if thumbnail.width > 1.48:
            thumbnail.set_width(1.48)
        if thumbnail.height > 0.72:
            thumbnail.set_height(0.72)
        output_label = Text("OUTPUT", color=MUTED, font_size=12, font=ANNOTATION_FONT)
        display_output = short_outputs.get(output, output) if compact else output
        output_parts = {
            "semantic centerline network": ("semantic centerline", "network"),
            "closed cross-section profiles": ("closed cross-section", "profiles"),
            "connected, welded triangle mesh": ("connected, welded", "triangle mesh"),
        }.get(display_output, (display_output,))
        output_text = VGroup(
            *[
                Text(
                    part,
                    color=TEXT,
                    font_size=13 if compact else 15,
                    font=ANNOTATION_FONT,
                )
                for part in output_parts
            ]
        ).arrange(DOWN, buff=0.04)
        # Fixed rows keep all six thumbnails and labels aligned despite
        # their different aspect ratios.
        stage.move_to(box.get_center() + UP * 0.91)
        title.move_to(box.get_center() + UP * 0.52)
        thumbnail.move_to(box.get_center() + UP * 0.03)
        output_label.move_to(box.get_center() + DOWN * 0.49)
        output_text.move_to(box.get_center() + DOWN * 0.80)
        contents = Group(stage, title, thumbnail, output_label, output_text)
        cards.append(Group(box, contents))
    card_row = Group(*cards).arrange(RIGHT, buff=0.25 if compact else 0.34)
    card_row.move_to((0.0, -0.42, 0.0))
    arrows = VGroup(
        *[
            Text("→", color=HIGHLIGHT, font_size=22, font=ANNOTATION_FONT).move_to(
                0.5 * (cards[index].get_right() + cards[index + 1].get_left())
            )
            for index in range(len(cards) - 1)
        ]
    )
    return card_row, arrows


def _event_interleave_flow() -> tuple[VGroup, tuple[VGroup, ...]]:
    """Show the implemented cave-surface-to-grounded-prop handoff."""

    specifications = (
        ("D", "cave surface", "#94a3b8"),
        ("E", "ground rocks", "#fb923c"),
        ("OUT", "cave + props", NETWORK),
    )
    nodes = []
    for stage, label, color in specifications:
        box = RoundedRectangle(
            width=2.35,
            height=0.52,
            corner_radius=0.10,
            color="#334155",
            fill_color="#10202e",
            fill_opacity=0.96,
            stroke_width=1.0,
        )
        text = Text(
            f"{stage} · {label}",
            color=color,
            font_size=15,
            font=ANNOTATION_FONT,
        )
        nodes.append(VGroup(box, text))
    row = VGroup(*nodes).arrange(RIGHT, buff=0.58)
    row.move_to((0.0, 2.08, 0.0))
    arrows = VGroup(
        *[
            Arrow(
                nodes[index].get_right() + RIGHT * 0.06,
                nodes[index + 1].get_left() + LEFT * 0.06,
                color=MUTED,
                stroke_width=1.4,
                max_tip_length_to_length_ratio=0.28,
            )
            for index in range(2)
        ]
    )
    return VGroup(row, arrows), tuple(nodes)


def _event_cross_section_parts(
    section: ManimSection,
    *,
    center_x: float = -3.15,
    maximum_width: float = 4.65,
) -> tuple[VGroup, tuple[tuple[Dot, np.ndarray], ...]]:
    """Build a profile-backed section with grounded rock and boulder props."""

    profile = np.asarray(section.profile_points_m, dtype=float)
    centered = profile - 0.5 * (np.min(profile, axis=0) + np.max(profile, axis=0))
    span = np.maximum(np.ptp(centered, axis=0), 1e-9)
    scale = min(maximum_width / span[0], 2.65 / span[1])
    center = np.asarray((center_x, -0.55, 0.0))
    points = np.column_stack(
        (
            centered[:, 0] * scale + center[0],
            centered[:, 1] * scale + center[1],
            np.zeros(len(centered)),
        )
    )
    outline = Polygon(
        *points,
        color=SECTIONS,
        fill_color="#10202e",
        fill_opacity=0.92,
        stroke_width=2.0,
    )
    left = float(np.min(points[:, 0]))
    right = float(np.max(points[:, 0]))
    bottom = float(np.min(points[:, 1]))
    floor_y = bottom + 0.34
    floor = Line(
        np.asarray((left + 0.42, floor_y, 0.0)),
        np.asarray((right - 0.42, floor_y, 0.0)),
        color="#94a3b8",
        stroke_width=2.0,
    )
    base_label = Text(
        "profile-derived cave section",
        color=MUTED,
        font_size=15,
        font=ANNOTATION_FONT,
    ).next_to(outline, DOWN, buff=0.12)
    base = VGroup(outline, floor, base_label)

    prop_specs = (
        (left + 1.05, 0.10, "#a16207"),
        (center[0] - 0.15, 0.16, "#92400e"),
        (right - 0.95, 0.12, "#b45309"),
    )
    props = []
    for x_value, radius, color in prop_specs:
        target = np.asarray((x_value, floor_y + radius, 0.0))
        start = target + UP * (1.20 + 0.30 * radius)
        dot = Dot(start, radius=radius, color=color)
        dot.set_stroke(color="#fed7aa", width=0.8, opacity=0.8)
        props.append((dot, target))

    return base, tuple(props)


def _geological_event_cross_section(
    section: ManimSection,
    *,
    compact: bool,
) -> VGroup:
    """Assemble the final Stage-E cross-section for overview use."""

    base, props = _event_cross_section_parts(section)
    for dot, target in props:
        dot.move_to(target)
    visual = VGroup(base, *[dot for dot, _target in props])
    if compact:
        visual.set_width(4.75)
    return visual


def _surface_preparation_parts(
    *,
    compact: bool,
) -> tuple[tuple[Group, ...], VGroup, VGroup]:
    """Build the implemented raw-mesh-to-portable-scene visual pipeline."""

    colors = ("#94a3b8", NETWORK, SECTIONS, "#4ade80")
    specifications = (
        ("Welded mesh", "shared topology", _mesh_patch_icon(rough=True)),
        ("Surface finish", "smooth + displacement", _mesh_patch_icon(rough=False)),
        ("Surface data", "UVs + tangents + PBR", _uv_pbr_icon()),
        ("Portable scene", "GLB · OBJ · USD", _portable_scene_icon()),
    )
    cards = []
    for (title, caption, icon), color in zip(specifications, colors, strict=True):
        box = RoundedRectangle(
            width=2.78,
            height=3.25,
            corner_radius=0.15,
            color=color,
            fill_color="#0d1c29",
            fill_opacity=0.95,
            stroke_width=1.3,
        )
        heading = Text(title, color=TEXT, font_size=19, font=ANNOTATION_FONT)
        caption_text = Text(caption, color=MUTED, font_size=14, font=ANNOTATION_FONT)
        contents = Group(heading, icon, caption_text).arrange(DOWN, buff=0.18)
        contents.move_to(box)
        cards.append(Group(box, contents))
    card_row = Group(*cards).arrange(RIGHT, buff=0.42)
    card_row.move_to((0.0, -0.25, 0.0))
    arrows = VGroup(
        *[
            Text("→", color=HIGHLIGHT, font_size=24, font=ANNOTATION_FONT).move_to(
                0.5 * (cards[index].get_right() + cards[index + 1].get_left())
            )
            for index in range(len(cards) - 1)
        ]
    )
    target_names = ("Blender", "UE5", "Unity", "Gazebo", "Omniverse")
    targets = VGroup(
        *[
            VGroup(
                RoundedRectangle(
                    width=1.55,
                    height=0.42,
                    corner_radius=0.08,
                    color="#334155",
                    fill_color="#10202e",
                    fill_opacity=0.95,
                    stroke_width=0.8,
                ),
                Text(name, color=TEXT, font_size=13, font=ANNOTATION_FONT),
            )
            for name in target_names
        ]
    ).arrange(RIGHT, buff=0.16)
    targets.move_to((0.0, -2.32, 0.0))
    if compact:
        combined = Group(card_row, arrows)
        combined.set_width(4.75)
        combined.move_to((0.0, 0.0, 0.0))
    return tuple(cards), arrows, targets


def _surface_preparation_strip(*, compact: bool) -> Group:
    cards, arrows, _targets = _surface_preparation_parts(compact=compact)
    strip = Group(*cards, arrows)
    if compact:
        strip.set_width(4.75)
    return strip


def _surface_channel_progress(
    channel_names: tuple[str, ...],
) -> tuple[VGroup, tuple[VGroup, ...]]:
    """Build the persistent channel-toggle rail for the Stage-F asset view."""

    colors = ("#94a3b8", NETWORK, SECTIONS, "#4ade80")
    nodes = []
    for name, color in zip(channel_names, colors, strict=True):
        box = RoundedRectangle(
            width=2.45,
            height=0.48,
            corner_radius=0.09,
            color="#334155",
            fill_color="#10202e",
            fill_opacity=0.96,
            stroke_width=0.9,
        )
        label = Text(name, color=color, font_size=14, font=ANNOTATION_FONT)
        nodes.append(VGroup(box, label))
    row = VGroup(*nodes).arrange(RIGHT, buff=0.42)
    row.move_to((0.0, 2.28, 0.0))
    arrows = VGroup(
        *[
            Text("→", color=MUTED, font_size=16, font=ANNOTATION_FONT).move_to(
                0.5 * (nodes[index].get_right() + nodes[index + 1].get_left())
            )
            for index in range(len(nodes) - 1)
        ]
    )
    return VGroup(row, arrows), tuple(nodes)


def _prepared_surface_view(
    sections: tuple[ManimSection, ...],
    yaw: float,
    channel: str,
) -> VGroup:
    """Project a rotating profile-backed tube with channel-specific styling."""

    ring_count = 14
    profiles = [
        _resample_closed_profile(section.profile_points_m, ring_count)
        for section in sections
    ]
    maximum = max(float(np.max(np.abs(profile))) for profile in profiles)
    profile_scale = 1.55 / max(maximum, 1e-9)
    axial_positions = np.linspace(-3.9, 3.9, len(profiles))
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    rings = []
    depths = []
    for ring_index, (axial, profile) in enumerate(
        zip(axial_positions, profiles, strict=True)
    ):
        lateral = profile[:, 0] * profile_scale
        vertical = profile[:, 1] * profile_scale
        if channel != "geometry":
            phase = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False)
            relief = 1.0 + 0.055 * np.sin(2.3 * axial + 3.0 * phase)
            lateral = lateral * relief
            vertical = vertical * relief
        rotated_x = cosine * axial - sine * lateral
        rotated_depth = sine * axial + cosine * lateral
        screen = np.column_stack(
            (
                1.08 * rotated_x,
                0.90 * vertical + 0.18 * rotated_depth - 0.35,
                np.zeros(ring_count),
            )
        )
        rings.append(screen)
        depths.append(rotated_depth)

    palettes = {
        "geometry": ("#64748b", "#94a3b8"),
        "displacement": ("#0891b2", "#22d3ee"),
        "normals + PBR": ("#7c3aed", "#60a5fa"),
        "packaged scene": ("#6b4f34", "#4ade80"),
    }
    dark, light = palettes[channel]
    cells = []
    for ring_index in range(len(rings) - 1):
        for point_index in range(ring_count):
            next_index = (point_index + 1) % ring_count
            points = (
                rings[ring_index][point_index],
                rings[ring_index + 1][point_index],
                rings[ring_index + 1][next_index],
                rings[ring_index][next_index],
            )
            depth = float(
                np.mean(
                    (
                        depths[ring_index][point_index],
                        depths[ring_index + 1][point_index],
                        depths[ring_index + 1][next_index],
                        depths[ring_index][next_index],
                    )
                )
            )
            color = light if (ring_index + point_index) % 2 else dark
            cells.append((depth, points, color))
    cells.sort(key=lambda item: item[0])
    fill_opacity = 0.18 if channel == "geometry" else 0.52
    faces = VGroup(
        *[
            Polygon(
                *points,
                color=color,
                fill_color=color,
                fill_opacity=fill_opacity,
                stroke_color="#0f172a",
                stroke_width=0.45,
                stroke_opacity=0.45,
            )
            for _depth, points, color in cells
        ]
    )
    wire = VGroup()
    for ring in rings:
        path = VMobject().set_points_as_corners(np.vstack((ring, ring[:1])))
        path.set_stroke(color=light, width=0.9, opacity=0.72)
        wire.add(path)
    for point_index in range(0, ring_count, 2):
        path = VMobject().set_points_as_corners(
            np.asarray([ring[point_index] for ring in rings])
        )
        path.set_stroke(color=light, width=0.75, opacity=0.58)
        wire.add(path)
    view = VGroup(faces, wire)
    if view.width > 9.25:
        view.set_width(9.25)
    if view.height > 4.25:
        view.set_height(4.25)
    view.move_to((0.0, -0.35, 0.0))
    return view


def _mesh_patch_icon(*, rough: bool) -> VGroup:
    """Create a small wire patch representing raw or prepared topology."""

    lines = []
    offsets = (0.14, -0.10, 0.08, -0.05) if rough else (0.04, -0.02, 0.02, -0.01)
    for row in range(4):
        y = 0.52 - row * 0.34
        points = []
        for column in range(5):
            x = -0.92 + column * 0.46
            points.append(np.asarray((x, y + offsets[(row + column) % 4], 0.0)))
        path = VMobject().set_points_as_corners(points)
        path.set_stroke(color="#94a3b8" if rough else NETWORK, width=1.5)
        lines.append(path)
    for column in range(5):
        points = []
        for row in range(4):
            y = 0.52 - row * 0.34
            x = -0.92 + column * 0.46
            points.append(np.asarray((x, y + offsets[(row + column) % 4], 0.0)))
        path = VMobject().set_points_as_corners(points)
        path.set_stroke(color="#64748b" if rough else "#22d3ee", width=1.1)
        lines.append(path)
    return VGroup(*lines)


def _uv_pbr_icon() -> VGroup:
    frame = Rectangle(width=1.90, height=1.25, color="#475569", stroke_width=1.0)
    islands = VGroup(
        Polygon(LEFT * 0.72 + UP * 0.36, LEFT * 0.15 + UP * 0.48, LEFT * 0.28, LEFT * 0.76, color=SECTIONS, fill_color=SECTIONS, fill_opacity=0.50),
        Polygon(RIGHT * 0.05 + DOWN * 0.46, RIGHT * 0.78 + DOWN * 0.30, RIGHT * 0.62 + UP * 0.12, RIGHT * 0.16, color=NETWORK, fill_color=NETWORK, fill_opacity=0.48),
    )
    swatches = VGroup(
        Rectangle(width=0.48, height=0.10, color="#92400e", fill_color="#92400e", fill_opacity=1.0),
        Rectangle(width=0.48, height=0.10, color="#60a5fa", fill_color="#60a5fa", fill_opacity=1.0),
        Rectangle(width=0.48, height=0.10, color="#9ca3af", fill_color="#9ca3af", fill_opacity=1.0),
    ).arrange(RIGHT, buff=0.08)
    swatches.next_to(frame, DOWN, buff=0.10)
    return VGroup(frame, islands, swatches)


def _portable_scene_icon() -> VGroup:
    document = Polygon(
        np.asarray((-0.72, -0.68, 0.0)),
        np.asarray((0.48, -0.68, 0.0)),
        np.asarray((0.72, -0.44, 0.0)),
        np.asarray((0.72, 0.68, 0.0)),
        np.asarray((-0.72, 0.68, 0.0)),
        color="#4ade80",
        fill_color="#123426",
        fill_opacity=0.92,
        stroke_width=1.4,
    )
    label = Text("GLB", color="#86efac", font_size=27, font=ANNOTATION_FONT)
    package = Text("mesh + maps", color=MUTED, font_size=12, font=ANNOTATION_FONT)
    package.next_to(label, DOWN, buff=0.08)
    return VGroup(document, label, package)


def _title(content: str) -> Text:
    title = Text(content, color=TEXT, font_size=34)
    title.to_edge(UP, buff=0.28)
    return title


def _prototype_data() -> ManimPrototypeData:
    return load_manim_prototype_data(
        _artifact_path("PLUME_VIDEO_NETWORK_ARTIFACT"),
        _artifact_path("PLUME_VIDEO_SECTION_ARTIFACT"),
        hero_segment_id=_optional_int("PLUME_VIDEO_HERO_SEGMENT"),
    )


def _stage_b_host_handoff(host: ManimHostData) -> Group:
    """Summarize the Stage-A substrate before Stage-B propagation begins."""

    cards = []
    for field_name, label, palette, color, unit in (
        ("elevation", "Terrain elevation", "terrain", "#38bdf8", "elevation (m)"),
        ("routing_cost", "Derived routing cost", "cost", HIGHLIGHT, "0–1 cost"),
    ):
        values = host.fields[field_name]
        image = ImageMobject(_field_rgba(values, palette))
        image.width = 4.15
        border = Rectangle(
            width=image.width + 0.07,
            height=image.height + 0.07,
            color=color,
            stroke_width=1.4,
        ).move_to(image)
        heading = VGroup(
            Text(label, color=TEXT, font_size=21, font=ANNOTATION_FONT),
            Text(
                "HOST FIELD" if field_name == "elevation" else "STAGE A OUTPUT",
                color=color,
                font_size=12,
                font=ANNOTATION_FONT,
            ),
        ).arrange(DOWN, buff=0.05)
        card = Group(
            heading,
            Group(image, border),
            _field_colorbar(values, palette, unit, width=3.85, compact=True),
        ).arrange(DOWN, buff=0.11)
        card.add(_map_sample_marker(image, compact=True))
        cards.append(card)

    arrow = VGroup(
        Text("→", color=HIGHLIGHT, font_size=34, font=ANNOTATION_FONT),
        Text("derive", color=MUTED, font_size=13, font=ANNOTATION_FONT),
    ).arrange(DOWN, buff=0.01)
    maps = Group(cards[0], arrow, cards[1]).arrange(RIGHT, buff=0.34)
    context = Text(
        "same spatial footprint · low cost indicates preferred propagation corridors",
        color=MUTED,
        font_size=16,
        font=ANNOTATION_FONT,
    )
    handoff = Group(maps, context).arrange(DOWN, buff=0.18)
    handoff.move_to((0.0, -0.38, 0.0))
    return handoff


def _routing_cost_overlay(host: ManimHostData, projector: PlanProjector) -> Group:
    """Resample Stage A routing cost into Stage B's exact rotated world frame."""

    pixel_width = 420
    pixel_height = 210
    screen_x = np.linspace(-5.65, 5.65, pixel_width)
    screen_y = np.linspace(2.825, -2.825, pixel_height)
    screen_grid_x, screen_grid_y = np.meshgrid(screen_x, screen_y)
    rotated = np.column_stack(
        (
            screen_grid_x.ravel() / projector.scale + projector.center[0],
            screen_grid_y.ravel() / projector.scale + projector.center[1],
        )
    )
    world = rotated @ projector.rotation
    world_x = world[:, 0].reshape((pixel_height, pixel_width))
    world_y = world[:, 1].reshape((pixel_height, pixel_width))

    x_axis = np.asarray(host.x_coords_m, dtype=float)
    y_axis = np.asarray(host.y_coords_m, dtype=float)
    values = np.asarray(host.fields["routing_cost"], dtype=float)
    if x_axis[0] > x_axis[-1]:
        x_axis = x_axis[::-1]
        values = values[:, ::-1]
    if y_axis[0] > y_axis[-1]:
        y_axis = y_axis[::-1]
        values = values[::-1, :]
    inside = (
        (world_x >= x_axis[0])
        & (world_x <= x_axis[-1])
        & (world_y >= y_axis[0])
        & (world_y <= y_axis[-1])
    )
    x_index = np.interp(world_x, x_axis, np.arange(len(x_axis), dtype=float))
    y_index = np.interp(world_y, y_axis, np.arange(len(y_axis), dtype=float))
    x0 = np.floor(x_index).astype(int)
    y0 = np.floor(y_index).astype(int)
    x1 = np.minimum(x0 + 1, len(x_axis) - 1)
    y1 = np.minimum(y0 + 1, len(y_axis) - 1)
    tx = x_index - x0
    ty = y_index - y0
    sampled = (
        (1.0 - tx) * (1.0 - ty) * values[y0, x0]
        + tx * (1.0 - ty) * values[y0, x1]
        + (1.0 - tx) * ty * values[y1, x0]
        + tx * ty * values[y1, x1]
    )
    rgba = _field_rgba(sampled, "cost")
    rgba[..., 3] = np.where(inside, 255, 0).astype(np.uint8)
    image = ImageMobject(rgba)
    image.stretch_to_fit_width(11.30)
    image.stretch_to_fit_height(5.65)
    image.move_to((0.0, -0.35, 0.0))
    image.set_opacity(0.24).set_z_index(-5)
    border = Rectangle(
        width=11.30,
        height=5.65,
        color="#475569",
        stroke_width=0.8,
        stroke_opacity=0.50,
    ).move_to(image).set_z_index(-4)
    label_box = RoundedRectangle(
        width=3.35,
        height=0.46,
        corner_radius=0.09,
        color=HIGHLIGHT,
        fill_color=BACKGROUND,
        fill_opacity=0.88,
        stroke_width=1.0,
    )
    label = Text(
        "Stage A routing cost · shared world coordinates",
        color=TEXT,
        font_size=13,
        font=ANNOTATION_FONT,
    )
    label_box.width = label.width + 0.30
    badge = VGroup(label_box, label).set_z_index(4)
    badge.move_to(
        image.get_corner(UP + LEFT)
        + RIGHT * (0.10 + 0.5 * badge.width)
        + DOWN * 0.25
    )
    return Group(image, border, badge)


def _host_field_focus(
    host: ManimHostData,
    field_name: str,
    label: str,
    description: str,
    palette: str,
    unit: str,
) -> Group:
    values = host.fields[field_name]
    image = ImageMobject(_field_rgba(values, palette))
    image.width = 4.40
    border = Rectangle(
        width=image.width + 0.08,
        height=image.height + 0.08,
        color="#475569",
        stroke_width=1.1,
    ).move_to(image)
    card = Group(
        VGroup(
            Text(label, color=TEXT, font_size=25, font=ANNOTATION_FONT),
            Text(description, color=MUTED, font_size=16, font=ANNOTATION_FONT),
        ).arrange(DOWN, buff=0.10),
        Group(image, border),
        Text(
            "same 1.8 × 1.5 km footprint",
            color=MUTED,
            font_size=15,
            font=ANNOTATION_FONT,
        ),
        _field_colorbar(values, palette, unit, width=3.70),
    ).arrange(DOWN, buff=0.16)
    card.move_to((3.35, -0.25, 0.0))
    marker = _map_sample_marker(image, compact=True)
    card.add(marker)
    return card


def _host_field_grid(
    host: ManimHostData,
    specifications: tuple[tuple[str, str, str, str, str], ...],
) -> Group:
    cards = []
    for field_name, label, _description, palette, unit in specifications:
        image = ImageMobject(_field_rgba(host.fields[field_name], palette))
        image.width = 2.12
        border = Rectangle(
            width=image.width + 0.05,
            height=image.height + 0.05,
            color=HIGHLIGHT if field_name == "routing_cost" else "#475569",
            stroke_width=1.5 if field_name == "routing_cost" else 0.9,
        ).move_to(image)
        heading = VGroup(
            Text(label, color=TEXT, font_size=16, font=ANNOTATION_FONT),
            Text(
                "SAVED OUTPUT" if field_name == "routing_cost" else "INPUT",
                color=HIGHLIGHT if field_name == "routing_cost" else MUTED,
                font_size=10,
                font=ANNOTATION_FONT,
            ),
        ).arrange(RIGHT, buff=0.10)
        scale = _field_colorbar(host.fields[field_name], palette, unit, width=1.95, compact=True)
        card = Group(
            heading,
            Group(image, border),
            scale,
        ).arrange(DOWN, buff=0.07)
        card.add(_map_sample_marker(image, compact=True))
        cards.append(card)
    rows = Group(
        Group(*cards[:2]).arrange(RIGHT, buff=0.20),
        Group(*cards[2:]).arrange(RIGHT, buff=0.20),
    ).arrange(DOWN, buff=0.18)
    heading = Text(
        "Host-field dashboard · one shared footprint",
        color=TEXT,
        font_size=21,
        font=ANNOTATION_FONT,
    )
    grid = Group(heading, rows).arrange(DOWN, buff=0.15)
    grid.move_to((3.35, -0.20, 0.0))
    return grid


def _routing_derivation(
    host: ManimHostData,
) -> tuple[Text, VGroup, VGroup, Group, Group]:
    """Explain the exact categories used to derive the saved routing cost."""

    terms = []
    for label in ("slope", "cover", "fracture", "capacity", "stability"):
        box = RoundedRectangle(
            width=1.28,
            height=0.48,
            corner_radius=0.10,
            color="#475569",
            fill_color="#132333",
            fill_opacity=0.95,
            stroke_width=1.0,
        )
        text = Text(label, color=TEXT, font_size=14, font=ANNOTATION_FONT)
        terms.append(VGroup(box, text))
    term_rows = VGroup(
        VGroup(*terms[:3]).arrange(RIGHT, buff=0.13),
        VGroup(*terms[3:]).arrange(RIGHT, buff=0.13),
    ).arrange(DOWN, buff=0.13)
    arrow = Text(
        "↓",
        color=HIGHLIGHT,
        font_size=25,
        font=ANNOTATION_FONT,
    )
    equation = Text(
        "weighted penalties are summed once",
        color=MUTED,
        font_size=15,
        font=ANNOTATION_FONT,
    )
    values = host.fields["routing_cost"]
    image = ImageMobject(_field_rgba(values, "cost"))
    image.width = 3.35
    border = Rectangle(
        width=image.width + 0.08,
        height=image.height + 0.08,
        color=HIGHLIGHT,
        stroke_width=1.6,
    ).move_to(image)
    result = Group(image, border)
    marker = _map_sample_marker(image, compact=True)
    result.add(marker)
    heading = Text(
        "Routing cost is derived",
        color=TEXT,
        font_size=25,
        font=ANNOTATION_FONT,
    )
    combine = VGroup(arrow, equation).arrange(DOWN, buff=0.01)
    scale = _field_colorbar(values, "cost", "0–1 cost", width=3.20)
    card = Group(
        heading,
        term_rows,
        combine,
        result,
        scale,
    ).arrange(DOWN, buff=0.14)
    card.move_to((3.35, -0.30, 0.0))
    return heading, term_rows, combine, result, scale


def _field_colorbar(
    values: np.ndarray,
    palette: str,
    unit: str,
    *,
    width: float,
    compact: bool = False,
) -> Group:
    """Build a projector-readable scalar legend with real artifact limits."""

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    gradient = np.linspace(minimum, maximum, 256, dtype=float)[None, :]
    gradient = np.repeat(gradient, 10, axis=0)
    image = ImageMobject(_field_rgba(gradient, palette))
    image.stretch_to_fit_width(width)
    image.stretch_to_fit_height(0.10 if compact else 0.15)
    border = Rectangle(
        width=width + 0.03,
        height=image.height + 0.03,
        color="#64748b",
        stroke_width=0.6,
    ).move_to(image)
    font_size = 10 if compact else 13
    decimals = 0 if unit in {"metres", "elevation (m)"} else 2
    low = Text(
        f"{minimum:.{decimals}f}",
        color=MUTED,
        font_size=font_size,
        font=ANNOTATION_FONT,
    )
    high = Text(
        f"{maximum:.{decimals}f}",
        color=MUTED,
        font_size=font_size,
        font=ANNOTATION_FONT,
    )
    unit_label = Text(
        unit,
        color=TEXT,
        font_size=font_size,
        font=ANNOTATION_FONT,
    )
    for label in (low, unit_label, high):
        label.next_to(image, DOWN, buff=0.04)
    low.align_to(image, LEFT)
    unit_label.set_x(image.get_center()[0])
    high.align_to(image, RIGHT)
    return Group(image, border, low, unit_label, high)


def _map_sample_marker(image: ImageMobject, *, compact: bool = False) -> VGroup:
    """Mark one stable normalized XY location on any Stage-A map."""

    u, v = 0.68, 0.38
    point = np.asarray(
        (
            image.get_left()[0] + u * image.width,
            image.get_bottom()[1] + v * image.height,
            0.0,
        )
    )
    arm = 0.09 if compact else 0.14
    return VGroup(
        Line(point + LEFT * arm, point + RIGHT * arm, color=SAMPLE_MARKER, stroke_width=2.0),
        Line(point + DOWN * arm, point + UP * arm, color=SAMPLE_MARKER, stroke_width=2.0),
        Dot(point, radius=0.025 if compact else 0.035, color=SAMPLE_MARKER),
    ).set_z_index(6)


def _map_context(host: ManimHostData, image: ImageMobject) -> VGroup:
    """Add scale, north, and footprint cues to the introductory plan map."""

    x_span = float(np.ptp(host.x_coords_m))
    y_span = float(np.ptp(host.y_coords_m))
    scale_length_m = 500.0
    scale_width = image.width * scale_length_m / max(x_span, 1e-9)
    scale_y = image.get_bottom()[1] + 0.22
    scale_x = image.get_left()[0] + 0.24
    bar = Line(
        np.asarray((scale_x, scale_y, 0.0)),
        np.asarray((scale_x + scale_width, scale_y, 0.0)),
        color=TEXT,
        stroke_width=4.0,
    )
    ticks = VGroup(
        Line(bar.get_start() + DOWN * 0.07, bar.get_start() + UP * 0.07, color=TEXT),
        Line(bar.get_end() + DOWN * 0.07, bar.get_end() + UP * 0.07, color=TEXT),
    )
    scale_label = Text(
        "500 m",
        color=TEXT,
        font_size=13,
        font=ANNOTATION_FONT,
    ).next_to(bar, UP, buff=0.04)
    north = Arrow(
        DOWN * 0.18,
        UP * 0.22,
        color=TEXT,
        stroke_width=2.0,
        max_tip_length_to_length_ratio=0.28,
    )
    north.move_to(image.get_corner(UP + LEFT) + RIGHT * 0.28 + DOWN * 0.34)
    north_label = Text("N", color=TEXT, font_size=14, font=ANNOTATION_FONT).next_to(
        north, UP, buff=0.02
    )
    footprint = Text(
        f"same footprint · {x_span / 1000:.1f} × {y_span / 1000:.1f} km",
        color=TEXT,
        font_size=13,
        font=ANNOTATION_FONT,
    )
    footprint.move_to(image.get_corner(DOWN + RIGHT) + LEFT * 1.25 + UP * 0.18)
    return VGroup(bar, ticks, scale_label, north, north_label, footprint).set_z_index(5)


def _map_wireframe(
    mesh_data: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]],
    image: ImageMobject,
) -> VGroup:
    """Overlay the exact downsampled mesh that will be lifted into relief."""

    x_nodes, y_nodes, _z_nodes, _colors = mesh_data
    lines = []
    for x_value in x_nodes:
        x = image.get_center()[0] + image.width * x_value / (2.0 * np.max(np.abs(x_nodes)))
        lines.append(
            Line(
                np.asarray((x, image.get_bottom()[1], 0.0)),
                np.asarray((x, image.get_top()[1], 0.0)),
                color=TEXT,
                stroke_width=0.65,
                stroke_opacity=0.48,
            )
        )
    for y_value in y_nodes:
        y = image.get_center()[1] + 0.5 * image.height * y_value
        lines.append(
            Line(
                np.asarray((image.get_left()[0], y, 0.0)),
                np.asarray((image.get_right()[0], y, 0.0)),
                color=TEXT,
                stroke_width=0.65,
                stroke_opacity=0.48,
            )
        )
    return VGroup(*lines).set_z_index(4)


def _prepare_elevation_mesh(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]]:
    """Downsample elevation and its cividis colors for responsive animation."""

    row_indices = np.linspace(0, values.shape[0] - 1, 13).round().astype(int)
    column_indices = np.linspace(0, values.shape[1] - 1, 21).round().astype(int)
    sampled = values[np.ix_(row_indices, column_indices)]
    lower, upper = np.percentile(values, (2.0, 98.0))
    normalized = np.clip((sampled - lower) / max(float(upper - lower), 1e-12), 0.0, 1.0)
    normalized -= 0.5
    aspect = values.shape[1] / values.shape[0]
    x_nodes = np.linspace(-aspect, aspect, sampled.shape[1])
    y_nodes = np.linspace(1.0, -1.0, sampled.shape[0])
    rgba = _field_rgba(values, "terrain")
    colors = []
    for row in range(len(row_indices) - 1):
        color_row = []
        for column in range(len(column_indices) - 1):
            pixel = rgba[
                (row_indices[row] + row_indices[row + 1]) // 2,
                (column_indices[column] + column_indices[column + 1]) // 2,
                :3,
            ]
            color_row.append(f"#{int(pixel[0]):02x}{int(pixel[1]):02x}{int(pixel[2]):02x}")
        colors.append(tuple(color_row))
    return x_nodes, y_nodes, normalized, tuple(colors)


def _elevation_surface(
    mesh_data: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]],
    *,
    yaw: float,
    tilt: float,
    relief: float,
    layout: float,
) -> VGroup:
    """Project the frozen elevation samples from top-down into rotating relief."""

    x_nodes, y_nodes, z_nodes, colors = mesh_data
    aspect = max(float(np.max(np.abs(x_nodes))), 1e-9)
    width = (1.0 - layout) * 6.20 + layout * 4.8
    center_x = (1.0 - layout) * -2.00 + layout * -3.55
    cosine_yaw = float(np.cos(yaw))
    sine_yaw = float(np.sin(yaw))
    cosine_tilt = float(np.cos(tilt))
    sine_tilt = float(np.sin(tilt))
    projected_half_width = abs(cosine_yaw) * aspect + abs(sine_yaw)
    scale = width / (2.0 * max(projected_half_width, 1e-9))

    def project(x_value: float, y_value: float, z_value: float) -> tuple[np.ndarray, float]:
        rotated_x = cosine_yaw * x_value - sine_yaw * y_value
        rotated_y = sine_yaw * x_value + cosine_yaw * y_value
        screen = np.asarray(
            (
                center_x + scale * rotated_x,
                -0.28
                - 0.70 * relief
                + scale
                * (cosine_tilt * rotated_y + 1.35 * relief * sine_tilt * z_value),
                0.0,
            )
        )
        depth = sine_tilt * rotated_y - cosine_tilt * relief * z_value
        return screen, float(depth)

    cells = []
    for row in range(len(y_nodes) - 1):
        for column in range(len(x_nodes) - 1):
            indices = (
                (row, column),
                (row, column + 1),
                (row + 1, column + 1),
                (row + 1, column),
            )
            projected = [
                project(x_nodes[column_index], y_nodes[row_index], z_nodes[row_index, column_index])
                for row_index, column_index in indices
            ]
            points = [item[0] for item in projected]
            mean_depth = float(np.mean([item[1] for item in projected]))
            cells.append((mean_depth, points, colors[row][column]))
    cells.sort(key=lambda item: item[0], reverse=True)
    return VGroup(
        *[
            Polygon(
                *points,
                color=color,
                fill_color=color,
                fill_opacity=1.0,
                stroke_color="#0f172a",
                stroke_width=0.35,
                stroke_opacity=0.22,
            )
            for _depth, points, color in cells
        ]
    )


def _elevation_sample_marker(
    mesh_data: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]],
    *,
    yaw: float,
    tilt: float,
    relief: float,
    layout: float,
) -> Dot:
    """Project the map marker onto the animated terrain surface."""

    x_nodes, y_nodes, z_nodes, _colors = mesh_data
    u, v = 0.68, 0.38
    x_value = float(np.min(x_nodes) + u * np.ptp(x_nodes))
    y_value = float(np.min(y_nodes) + v * np.ptp(y_nodes))
    x_index = int(np.argmin(np.abs(x_nodes - x_value)))
    y_index = int(np.argmin(np.abs(y_nodes - y_value)))
    z_value = float(z_nodes[y_index, x_index])
    aspect = max(float(np.max(np.abs(x_nodes))), 1e-9)
    width = (1.0 - layout) * 6.20 + layout * 4.8
    center_x = (1.0 - layout) * -2.00 + layout * -3.55
    cosine_yaw = float(np.cos(yaw))
    sine_yaw = float(np.sin(yaw))
    cosine_tilt = float(np.cos(tilt))
    sine_tilt = float(np.sin(tilt))
    projected_half_width = abs(cosine_yaw) * aspect + abs(sine_yaw)
    scale = width / (2.0 * max(projected_half_width, 1e-9))
    rotated_x = cosine_yaw * x_value - sine_yaw * y_value
    rotated_y = sine_yaw * x_value + cosine_yaw * y_value
    point = np.asarray(
        (
            center_x + scale * rotated_x,
            -0.28
            - 0.70 * relief
            + scale * (cosine_tilt * rotated_y + 1.35 * relief * sine_tilt * z_value),
            0.0,
        )
    )
    return Dot(point, radius=0.075, color=SAMPLE_MARKER).set_z_index(7)


def _field_rgba(values: np.ndarray, palette: str) -> np.ndarray:
    from matplotlib import colormaps

    palette_names = {
        "terrain": "cividis",
        "cover": "viridis",
        "fracture": "magma",
        "stability": "YlGn",
        "cost": "magma_r",
    }
    lower, upper = np.percentile(values, (2.0, 98.0))
    normalized = np.clip((values - lower) / max(float(upper - lower), 1e-12), 0.0, 1.0)
    rgba = colormaps[palette_names[palette]](normalized)
    return np.rint(255.0 * rgba).astype(np.uint8)


def _flow_animations(
    segments: tuple[ManimSegment, ...],
    paths: tuple[VMobject, ...],
    *,
    total_time: float,
) -> tuple[Succession, ...]:
    starts = np.asarray([segment.age_start_s for segment in segments], dtype=float)
    ends = np.asarray(
        [max(segment.age_start_s, segment.age_end_s) for segment in segments],
        dtype=float,
    )
    lower = float(np.min(starts))
    upper = float(np.max(ends))
    span = max(upper - lower, 1e-9)
    scheduled = []
    for segment, path in zip(segments, paths, strict=True):
        start_time = total_time * max(segment.age_start_s - lower, 0.0) / span
        duration = total_time * max(segment.age_end_s - segment.age_start_s, 0.0) / span
        duration = max(duration, 0.32)
        animations = []
        if start_time > 1e-6:
            animations.append(Wait(start_time))
        animations.append(Create(path, run_time=duration))
        scheduled.append(Succession(*animations))
    return tuple(scheduled)


def _segment_type_color(kind: str) -> str:
    """Return the stable Stage-B color assigned to a semantic segment role."""

    return SEGMENT_TYPE_COLORS.get(kind, MUTED)


def _segment_type_legend(segments: tuple[ManimSegment, ...]) -> VGroup:
    """Build a compact two-column legend containing only roles in the artifact."""

    kinds = tuple(kind for kind in SEGMENT_TYPE_COLORS if any(s.kind == kind for s in segments))
    unknown_kinds = tuple(
        sorted({segment.kind for segment in segments}.difference(SEGMENT_TYPE_COLORS))
    )
    entries = []
    for kind in (*kinds, *unknown_kinds):
        swatch = Line(LEFT * 0.16, RIGHT * 0.16, color=_segment_type_color(kind), stroke_width=5)
        label = Text(
            kind.replace("_", " "),
            color=TEXT,
            font_size=15,
            font=ANNOTATION_FONT,
        )
        entries.append(VGroup(swatch, label).arrange(RIGHT, buff=0.12))

    split = (len(entries) + 1) // 2
    columns = VGroup(
        VGroup(*entries[:split]).arrange(DOWN, aligned_edge=LEFT, buff=0.10),
        VGroup(*entries[split:]).arrange(DOWN, aligned_edge=LEFT, buff=0.10),
    ).arrange(RIGHT, aligned_edge=UP, buff=0.35)
    heading = Text(
        "segment role",
        color=MUTED,
        font_size=15,
        font=ANNOTATION_FONT,
    )
    legend = VGroup(heading, columns).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    legend.to_corner(DOWN + RIGHT, buff=0.42).shift(UP * 0.52)
    return legend


def _stage_d_step(label: str) -> Text:
    step = Text(label, color=TEXT, font_size=22, font=ANNOTATION_FONT)
    step.move_to((0.0, 1.92, 0.0))
    return step


def _stage_d_side_note(lines: tuple[str, ...], color: str) -> VGroup:
    note = VGroup(
        *[Text(line, color=color, font_size=15, font=ANNOTATION_FONT) for line in lines]
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    note.to_edge(RIGHT, buff=0.42).set_y(-2.15)
    return note


def _stage_d_progress() -> tuple[VGroup, tuple[VGroup, ...]]:
    """Keep the complete profiles-to-welded-mesh sequence visible."""

    specifications = (
        ("profiles", SECTIONS, "○"),
        ("density", SAMPLE_MARKER, "·"),
        ("signs", HIGHLIGHT, "±"),
        ("surface", NETWORK, "△"),
        ("welded", "#f472b6", "#"),
    )
    nodes = []
    for label, color, glyph in specifications:
        box = RoundedRectangle(
            width=1.86,
            height=0.48,
            corner_radius=0.09,
            color="#334155",
            fill_color="#10202e",
            fill_opacity=0.96,
            stroke_width=0.9,
        )
        icon = Text(glyph, color=color, font_size=18, font=ANNOTATION_FONT)
        text = Text(label, color=TEXT, font_size=13, font=ANNOTATION_FONT)
        contents = VGroup(icon, text).arrange(RIGHT, buff=0.10)
        nodes.append(VGroup(box, contents))
    row = VGroup(*nodes).arrange(RIGHT, buff=0.36)
    row.move_to((0.0, 2.43, 0.0))
    arrows = VGroup(
        *[
            Text("→", color=MUTED, font_size=16, font=ANNOTATION_FONT).move_to(
                0.5 * (nodes[index].get_right() + nodes[index + 1].get_left())
            )
            for index in range(len(nodes) - 1)
        ]
    )
    return VGroup(row, arrows), tuple(nodes)


def _profile_density_slice(section: ManimSection) -> VGroup:
    """Show one section profile sampled by a signed-density voxel slice."""

    profile = np.asarray(section.profile_points_m, dtype=float)
    centered = profile - 0.5 * (np.min(profile, axis=0) + np.max(profile, axis=0))
    span = np.maximum(np.ptp(centered, axis=0), 1e-9)
    scale = min(4.2 / span[0], 3.1 / span[1])
    polygon = centered * scale
    grid_x = np.linspace(-2.7, 2.7, 19)
    grid_y = np.linspace(-1.75, 1.75, 13)
    grid_points = np.asarray([(x, y) for y in grid_y for x in grid_x], dtype=float)
    inside = _points_in_polygon(grid_points, polygon)
    dots = VGroup(
        *[
            Dot(
                np.asarray((point[0], point[1] - 0.25, 0.0)),
                radius=0.026,
                color=SAMPLE_MARKER if is_inside else "#475569",
            ).set_opacity(0.90 if is_inside else 0.38)
            for point, is_inside in zip(grid_points, inside, strict=True)
        ]
    )
    outline_points = np.column_stack(
        (
            np.vstack((polygon, polygon[:1]))[:, 0],
            np.vstack((polygon, polygon[:1]))[:, 1] - 0.25,
            np.zeros(len(polygon) + 1),
        )
    )
    outline = VMobject().set_points_as_corners(outline_points)
    outline.set_fill(opacity=0.0)
    outline.set_stroke(color=SECTIONS, width=3.0, opacity=1.0)
    return VGroup(dots, outline)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd containment for the explanatory density slice."""

    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        crossing = (y0 > y) != (y1 > y)
        x_crossing = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
        inside ^= crossing & (x < x_crossing)
        x0, y0 = x1, y1
    return inside


def _density_key() -> VGroup:
    entries = VGroup(
        VGroup(
            Dot(radius=0.055, color=SAMPLE_MARKER),
            Text("density ≥ 0 · carved", color=TEXT, font_size=16, font=ANNOTATION_FONT),
        ).arrange(RIGHT, buff=0.12),
        VGroup(
            Dot(radius=0.055, color="#475569"),
            Text("density < 0 · rock", color=MUTED, font_size=16, font=ANNOTATION_FONT),
        ).arrange(RIGHT, buff=0.12),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    entries.to_corner(DOWN + RIGHT, buff=0.55).shift(UP * 0.62)
    return entries


def _marching_cube_case(
) -> tuple[VGroup, VGroup, VGroup, VGroup, VGroup, VGroup, VGroup]:
    """Construct a projected cube whose zero-level surface is a two-triangle patch."""

    corners_xyz = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
        dtype=float,
    )
    density_values = np.asarray((-0.9, -0.4, -0.7, -1.0, 0.8, 0.5, 0.9, 0.6))
    projected = np.asarray([_project_cube_point(point) for point in corners_xyz])
    edge_indices = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    faces = VGroup(
        Polygon(
            projected[0], projected[1], projected[5], projected[4],
            color="#0e7490", fill_color="#0e7490", fill_opacity=0.10,
            stroke_opacity=0.0,
        ),
        Polygon(
            projected[1], projected[2], projected[6], projected[5],
            color="#1d4ed8", fill_color="#1d4ed8", fill_opacity=0.08,
            stroke_opacity=0.0,
        ),
        Polygon(
            projected[4], projected[5], projected[6], projected[7],
            color="#38bdf8", fill_color="#38bdf8", fill_opacity=0.08,
            stroke_opacity=0.0,
        ),
    )
    rear_edge_indices = {2, 3, 6, 7, 11}
    edges = VGroup(
        *[
            (
                DashedLine(
                    projected[start],
                    projected[end],
                    color="#475569",
                    stroke_width=1.5,
                    dash_length=0.10,
                    dashed_ratio=0.55,
                )
                if index in rear_edge_indices
                else Line(
                    projected[start],
                    projected[end],
                    color="#94a3b8",
                    stroke_width=2.2,
                )
            )
            for index, (start, end) in enumerate(edge_indices)
        ]
    )
    dots = VGroup(
        *[
            Dot(
                point,
                radius=0.075,
                color=SAMPLE_MARKER if index >= 4 else "#64748b",
            ).set_stroke(color=TEXT, width=1.0)
            for index, point in enumerate(projected)
        ]
    )
    labels = VGroup(
        *[
            Text(
                f"{density_values[index]:+.1f}",
                color=SAMPLE_MARKER if index >= 4 else MUTED,
                font_size=15,
                font=ANNOTATION_FONT,
            ).next_to(dot, UP if index >= 4 else DOWN, buff=0.05)
            for index, dot in enumerate(dots)
        ]
    )
    vertical_edges = ((0, 4), (1, 5), (2, 6), (3, 7))
    sign_edges = VGroup(
        *[
            Line(
                projected[start],
                projected[end],
                color=HIGHLIGHT,
                stroke_width=4.0,
            )
            for start, end in vertical_edges
        ]
    )
    crossing_points = []
    for start, end in vertical_edges:
        start_density = density_values[start]
        end_density = density_values[end]
        fraction = -start_density / (end_density - start_density)
        crossing_xyz = corners_xyz[start] + fraction * (corners_xyz[end] - corners_xyz[start])
        crossing_points.append(_project_cube_point(crossing_xyz))
    crossing_points = np.asarray(crossing_points)
    crossings = VGroup(
        *[
            Dot(point, radius=0.06, color=HIGHLIGHT).set_stroke(color=TEXT, width=1.0)
            for point in crossing_points
        ]
    )
    triangles = VGroup(
        Polygon(
            crossing_points[0], crossing_points[1], crossing_points[2],
            color=NETWORK, fill_color=NETWORK, fill_opacity=0.42, stroke_width=2.0,
        ),
        Polygon(
            crossing_points[0], crossing_points[2], crossing_points[3],
            color="#60a5fa", fill_color="#60a5fa", fill_opacity=0.42, stroke_width=2.0,
        ),
    )
    return faces, edges, dots, labels, sign_edges, crossings, triangles


def _project_cube_point(point: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    x_value, y_value, z_value = np.asarray(point, dtype=float)
    return np.asarray(
        (
            2.30 * x_value - 1.65 * y_value + 0.18 * z_value - 0.32,
            0.85 * x_value + 0.65 * y_value + 2.50 * z_value - 2.85,
            0.0,
        )
    )


def _section_tube_mesh(
    sections: tuple[ManimSection, ...],
) -> tuple[VGroup, VGroup, VGroup]:
    """Build a perspective wireframe tube from the actual frozen section profiles."""

    ring_count = 12
    profiles = [_resample_closed_profile(section.profile_points_m, ring_count) for section in sections]
    maximum = max(float(np.max(np.abs(profile))) for profile in profiles)
    profile_scale = 1.35 / max(maximum, 1e-9)
    axial_positions = np.linspace(-4.2, 4.2, len(profiles))
    rings: list[np.ndarray] = []
    for index, (axial, profile) in enumerate(zip(axial_positions, profiles, strict=True)):
        depth = profile[:, 0] * profile_scale
        height = profile[:, 1] * profile_scale
        axis_height = 0.28 - 0.56 * index / max(len(profiles) - 1, 1)
        rings.append(
            np.column_stack(
                (
                    axial + 0.38 * depth,
                    axis_height + height + 0.13 * depth - 0.25,
                    np.zeros(ring_count),
                )
            )
        )

    faces = VGroup()
    for ring_index in range(len(rings) - 1):
        for point_index in range(ring_count):
            next_index = (point_index + 1) % ring_count
            quad = (
                rings[ring_index][point_index],
                rings[ring_index + 1][point_index],
                rings[ring_index + 1][next_index],
                rings[ring_index][next_index],
            )
            colors = ("#0e7490", "#2563eb") if (ring_index + point_index) % 2 else ("#0891b2", "#3b82f6")
            faces.add(
                Polygon(
                    quad[0], quad[1], quad[2],
                    color=colors[0], fill_color=colors[0], fill_opacity=0.30, stroke_opacity=0.0,
                ),
                Polygon(
                    quad[0], quad[2], quad[3],
                    color=colors[1], fill_color=colors[1], fill_opacity=0.30, stroke_opacity=0.0,
                ),
            )

    wire = VGroup()
    for ring in rings:
        for point_index in range(ring_count):
            wire.add(
                Line(
                    ring[point_index],
                    ring[(point_index + 1) % ring_count],
                    color="#94a3b8",
                    stroke_width=1.0,
                )
            )
    for ring_index in range(len(rings) - 1):
        for point_index in range(0, ring_count, 2):
            wire.add(
                Line(
                    rings[ring_index][point_index],
                    rings[ring_index + 1][point_index],
                    color="#94a3b8",
                    stroke_width=1.0,
                )
            )

    seam_ring = rings[len(rings) // 2]
    seam = VGroup(
        *[
            Line(
                seam_ring[index],
                seam_ring[(index + 1) % ring_count],
                color="#f472b6",
                stroke_width=4.0,
            )
            for index in range(ring_count)
        ]
    )
    return faces, wire, seam


def _resample_closed_profile(profile: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(profile, dtype=float)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    indices = np.linspace(0, len(points), count, endpoint=False)
    lower = np.floor(indices).astype(int) % len(points)
    upper = (lower + 1) % len(points)
    fraction = indices - np.floor(indices)
    return (1.0 - fraction[:, None]) * points[lower] + fraction[:, None] * points[upper]


def _passage_envelopes(
    data: ManimPrototypeData,
    projector: PlanProjector,
) -> VGroup:
    section_lookup: dict[int, list[ManimSection]] = {}
    for section in data.sections:
        section_lookup.setdefault(section.segment_id, []).append(section)
    return VGroup(
        *(
            envelope
            for segment in data.segments
            if (
                envelope := _passage_envelope(
                    tuple(section_lookup.get(segment.segment_id, ())),
                    projector,
                )
            )
            is not None
        )
    )


class PlanProjector:
    """Map physical XY coordinates into a stable Manim plan-view frame."""

    def __init__(
        self,
        segments: tuple[ManimSegment, ...],
        *,
        rotation_radians: float = 0.0,
    ) -> None:
        points = np.vstack([segment.points_xyz_m[:, :2] for segment in segments])
        cosine = float(np.cos(rotation_radians))
        sine = float(np.sin(rotation_radians))
        self.rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=float)
        rotated = points @ self.rotation.T
        lower = np.min(rotated, axis=0)
        upper = np.max(rotated, axis=0)
        self.center = 0.5 * (lower + upper)
        span = np.maximum(upper - lower, 1e-9)
        self.scale = min(11.3 / span[0], 5.65 / span[1])

    def points(self, points_xyz_m: np.ndarray) -> np.ndarray:
        rotated = points_xyz_m[:, :2] @ self.rotation.T
        xy = (rotated - self.center) * self.scale
        return np.column_stack((xy[:, 0], xy[:, 1] - 0.35, np.zeros(len(xy))))

def _segment_path(
    segment: ManimSegment,
    projector: PlanProjector,
    color: str,
    width: float,
) -> VMobject:
    path = VMobject()
    points = projector.points(segment.points_xyz_m)
    path.set_points_as_corners(points)
    path.set_fill(opacity=0.0)
    path.set_stroke(color=color, width=width, opacity=1.0)
    return path


def _passage_envelope(
    sections: tuple[ManimSection, ...],
    projector: PlanProjector,
) -> VMobject | None:
    """Project the physical Stage-C widths into a filled plan-view ribbon."""

    ordered = tuple(sorted(sections, key=lambda item: item.arc_length_m))
    if len(ordered) < 2:
        return None
    centers = np.asarray([section.center_xyz_m for section in ordered], dtype=float)
    normals_xy = np.asarray([section.normal_xyz[:2] for section in ordered], dtype=float)
    previous: np.ndarray | None = None
    for index, normal in enumerate(normals_xy):
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            start = centers[max(0, index - 1), :2]
            end = centers[min(len(centers) - 1, index + 1), :2]
            tangent = end - start
            normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
            length = float(np.linalg.norm(normal))
        if length < 1e-8:
            return None
        normal = normal / length
        if previous is not None and float(np.dot(normal, previous)) < 0.0:
            normal = -normal
        normals_xy[index] = normal
        previous = normal
    half_widths = 0.5 * np.asarray([section.width_m for section in ordered], dtype=float)
    left = centers.copy()
    right = centers.copy()
    left[:, :2] += normals_xy * half_widths[:, None]
    right[:, :2] -= normals_xy * half_widths[:, None]
    boundary = np.vstack(
        (
            projector.points(left),
            projector.points(right)[::-1],
            projector.points(left[:1]),
        )
    )
    envelope = VMobject()
    envelope.set_points_as_corners(boundary)
    envelope.set_fill(GEOMETRY, opacity=0.52)
    envelope.set_stroke(color="#94a3b8", width=1.15, opacity=0.92)
    return envelope


def _section_panel(
    sections: tuple[ManimSection, ...],
    *,
    width: float = 5.25,
    height: float = 1.65,
    caption: str = "Same branch · sampled profiles",
) -> tuple[VGroup, tuple[VMobject, ...]]:
    frame = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.12,
        color="#334155",
        stroke_width=1.2,
        fill_color=BACKGROUND,
        fill_opacity=0.92,
    )
    frame.to_corner(DOWN + RIGHT, buff=0.42)
    label = Text(
        caption,
        color=MUTED,
        font_size=16,
        font=ANNOTATION_FONT,
    )
    label.next_to(frame.get_top(), DOWN, buff=0.12)
    profiles = []
    if sections:
        centers = np.linspace(
            frame.get_left()[0] + 0.62,
            frame.get_right()[0] - 0.62,
            len(sections),
        )
        for center_x, section in zip(centers, sections, strict=True):
            profile = np.asarray(section.profile_points_m, dtype=float)
            scale = min(0.72 / max(np.ptp(profile[:, 0]), 1e-9), 0.86 / max(np.ptp(profile[:, 1]), 1e-9))
            points = np.column_stack(
                (
                    (profile[:, 0] - np.mean(profile[:, 0])) * scale + center_x,
                    (profile[:, 1] - np.mean(profile[:, 1])) * scale + frame.get_center()[1] - 0.08,
                    np.zeros(len(profile)),
                )
            )
            contour = VMobject(color=SECTIONS, stroke_width=2.0)
            contour.set_points_as_corners(np.vstack((points, points[0])))
            profiles.append(contour)
    ticks = VGroup(
        *[
            Line(
                [profile.get_center()[0], frame.get_bottom()[1] + 0.10, 0],
                [profile.get_center()[0], frame.get_bottom()[1] + 0.20, 0],
                color=MUTED,
                stroke_width=1.0,
            )
            for profile in profiles
        ]
    )
    return VGroup(frame, label, ticks), tuple(profiles)


def _section_progress_badge(alpha: float) -> VGroup:
    """Label the current normalized position along the selected branch."""

    box = RoundedRectangle(
        width=1.38,
        height=0.42,
        corner_radius=0.09,
        color=SAMPLE_MARKER,
        fill_color="#12231f",
        fill_opacity=0.96,
        stroke_width=1.1,
    )
    label = Text(
        f"s / L = {alpha:.2f}",
        color=SAMPLE_MARKER,
        font_size=15,
        font=ANNOTATION_FONT,
    )
    return VGroup(box, label)


def _artifact_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(
            f"{variable} is not set. Use scripts/render_manim_video.py to render this scene."
        )
    return Path(value).resolve()


def _optional_int(variable: str) -> int | None:
    value = os.environ.get(variable)
    return None if value in (None, "") else int(value)
