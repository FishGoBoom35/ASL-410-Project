"""Build the baseline-model dataset from every video under data/ASL Data Collection.

Returns:
    X_df    one row per clip: clip_id, label, signer, path, then feature columns f0, f1, ...
    failed  list of (path, reason) for clips that could not be processed
"""
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

COLUMNS = ["frame", "hand", "landmark", "x", "y", "z"]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
N_HANDS, N_LANDMARKS = 2, 21


# ---------------------------------------------------------------- extraction
def extract(video_path, model_path="models/hand_landmarker.task"):
    """Run MediaPipe hand tracking on one video.
    Returns a long-format DataFrame: frame, hand, landmark, x, y, z."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS) or 30  # fall back if fps is missing
    rows = []
    frame_number = 0

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = video.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(frame_number * 1000 / fps)
            result = detector.detect_for_video(mp_image, timestamp)

            for hand_index, hand in enumerate(result.hand_landmarks):
                for landmark_id, point in enumerate(hand):
                    rows.append((frame_number, hand_index, landmark_id, point.x, point.y, point.z))

            frame_number += 1

    video.release()
    return pd.DataFrame(rows, columns=COLUMNS)


# ------------------------------------------------------------------ cleaning
def df_to_array(df):
    """Long format -> (T, 2, 21, 3) array. NaN wherever a hand wasn't detected."""
    if df.empty:
        raise ValueError("No landmarks in this clip (no hands detected).")
    first, last = df["frame"].min(), df["frame"].max()
    arr = np.full((last - first + 1, N_HANDS, N_LANDMARKS, 3), np.nan)
    arr[df["frame"].values - first, df["hand"].values, df["landmark"].values] = df[["x", "y", "z"]].values
    return arr


def fill_gaps(arr, min_frac=0.3):
    """Interpolate short detection gaps. A hand seen in fewer than min_frac of the
    frames is treated as 'not used' and set to zeros."""
    out = arr.copy()
    t = np.arange(len(out))
    for h in range(N_HANDS):
        present = ~np.isnan(out[:, h, 0, 0])  # all 21 landmarks appear together
        if present.mean() < min_frac:
            out[:, h] = 0.0
            continue
        for l in range(N_LANDMARKS):
            for c in range(3):
                out[:, h, l, c] = np.interp(t, t[present], out[present, h, l, c])
    return out


def normalize(arr):
    """Per frame and hand: hand shape relative to the wrist, scaled by hand size
    (wrist -> middle-finger base). Wrist position (x, y) is kept as separate features
    because where the hand is matters for signs. Returns (T, 130)."""
    T = len(arr)
    rel = arr - arr[:, :, 0:1, :]
    size = np.linalg.norm(arr[:, :, 9, :2] - arr[:, :, 0, :2], axis=-1)  # (T, 2)
    size = np.where(size > 1e-6, size, 1.0)
    rel = rel / size[:, :, None, None]
    wrist_pos = arr[:, :, 0, :2]                                          # (T, 2, 2)
    return np.concatenate([rel.reshape(T, -1), wrist_pos.reshape(T, -1)], axis=1)


def resample(seq, n_frames=30):
    """Stretch/squeeze a (T, F) sequence to exactly n_frames."""
    old, new = np.linspace(0, 1, len(seq)), np.linspace(0, 1, n_frames)
    return np.stack([np.interp(new, old, seq[:, i]) for i in range(seq.shape[1])], axis=1)


def clip_to_row(df, n_frames=30, mode="summary", **meta):
    """One-row DataFrame for the baseline models.

    mode="summary": mean, std, min, max of each feature over time (4 * 130 = 520 values)
    mode="flatten": every frame's features side by side (n_frames * 130 values)
    **meta: extra columns placed first, e.g. clip_id=..., label=..., signer=...
    """
    seq = resample(normalize(fill_gaps(df_to_array(df))), n_frames)
    if mode == "summary":
        feats = np.concatenate([seq.mean(0), seq.std(0), seq.min(0), seq.max(0)])
    elif mode == "flatten":
        feats = seq.ravel()
    else:
        raise ValueError("mode must be 'summary' or 'flatten'")
    feat_df = pd.DataFrame([feats], columns=[f"f{i}" for i in range(len(feats))])
    return pd.concat([pd.DataFrame([meta]), feat_df], axis=1) if meta else feat_df


# ------------------------------------------------------------ whole dataset
def build_dataset(n_frames=30, mode="summary"):
    data_dir = Path('data')
    paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    print(f"Found {len(paths)} videos under {data_dir}")

    rows, failed = [], []
    for i, path in enumerate(paths, 1):
        signer = path.parent.parent.name.replace("ASL_", "")  # ASL_CIAN -> CIAN
        label = path.parent.name.strip().lower()              # Book -> book
        clip_id = f"{signer}_{path.stem}"                     # CIAN_book1 (unique across signers)

        try:
            landmarks = extract(path, 'models/hand_landmarker.task')
            rows.append(
                clip_to_row(
                    landmarks, n_frames=n_frames, mode=mode,
                    clip_id=clip_id, label=label, signer=signer, path=str(path),
                )
            )
        except (RuntimeError, ValueError) as err:
            failed.append((str(path), str(err)))
            print(f"[{i}/{len(paths)}] SKIPPED {clip_id}: {err}")
            continue

        print(f"[{i}/{len(paths)}] {clip_id} ({label})")

    X_df = pd.concat(rows, ignore_index=True)
    print(f"Done: {len(X_df)} clips kept, {len(failed)} skipped")
    return X_df, failed

X_df, failed = build_dataset()

X_df.to_csv('dataset.csv', index=False)