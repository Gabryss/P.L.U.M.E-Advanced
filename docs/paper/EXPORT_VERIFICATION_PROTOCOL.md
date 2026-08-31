# Manual export verification protocol

Record the package, commit, resolved-config hash, seed, application version,
operator, date, and screenshots. Emission is not a successful engine import.

For Blender, Gazebo, Unity, UE5, and Omniverse:

1. Generate the frozen representative seed and follow generated import notes.
2. Confirm the documented transform and units with a 1 m reference cube.
3. Record canonical and imported bounding boxes.
4. Confirm visual assets and material/texture bindings.
5. Confirm separate collision geometry when requested.
6. Drop a dynamic 0.5 m cube and verify it rests on the floor without a
   metre-scale offset.
7. Save `<target>_<version>_<seed>_{scale,visual,collision}.png`.
8. Preserve pass/fail and the exact failure reason.

The paper may say packages were emitted for five targets after automatic
checks. It may report manual imports only for completed records, and must not
claim identical rendering, physics, or sensor behavior.
