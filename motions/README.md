# Reference Motion

## `right_kick.npz`

Default right-foot push-kick reference motion used in RoboNaldo training.

Source:

- Retarget: GVHMR+GMR.
- Artifact entry: `motion.npz`

Format:

- 612 frames
- 50 Hz
- `joint_pos` / `joint_vel`: `(612, 29)`
- `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w`: `(612, 30, 3)`
- `body_quat_w`: `(612, 30, 4)` in `wxyz`

To replace it, overwrite `motions/right_kick.npz` with another
RoboNaldo-format NPZ. Optional upload:

```bash
python scripts/upload_npz.py \
  --artifact_path motions/right_kick.npz \
  --entity <entity> \
  --name right_kick
```

The legacy `right_kick_reference.csv` is kept only as source/reference data for
custom conversion experiments.
