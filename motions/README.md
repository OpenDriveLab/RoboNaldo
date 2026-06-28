# Reference Motions

## `right_kick_reference.csv`

Open-source right-foot kick reference motion used in RoboNaldo.

Source:

- Retarget: GVHMR+GMR.
- Artifact entry: `motion.npz`

Format:

- 612 frames
- 50 Hz
- 36 comma-separated columns per row
- Columns: root position `(x, y, z)`, root quaternion `(x, y, z, w)`, then 29
  Unitree G1 joint positions in the order used by `scripts/csv_to_npz.py`

Convert to training NPZ:

```bash
python scripts/csv_to_npz.py \
  --input_file motions/right_kick_reference.csv \
  --input_fps 50 \
  --output_name right_kick \
  --headless
```
