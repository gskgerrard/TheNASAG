import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURES = ['latitude', 'longitude', 'bright_ti4', 'scan', 'track', 'bright_ti5']
TARGET = 'frp'


def generate_synthetic_environment_data(timesteps=60, lat=64, lon=64, random_state=42):
    """Create a small synthetic spatiotemporal demo grid.

    These generated CO/IR-like channels are not NASA observations and must not
    be used as real environmental data.
    The output shape is (time, latitude, longitude, 2).
    """
    if any(not isinstance(size, (int, np.integer)) or size < 1
           for size in (timesteps, lat, lon)):
        raise ValueError("timesteps, lat, and lon must be positive integers")
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2 to build sequences")

    rng = np.random.default_rng(random_state)
    y_grid, x_grid = np.mgrid[-1:1:complex(lat), -1:1:complex(lon)]
    frames = np.empty((timesteps, lat, lon, 2), dtype=np.float32)

    for step in range(timesteps):
        phase = 2 * np.pi * step / max(timesteps, 1)
        plume_x, plume_y = 0.38 * np.sin(phase), 0.32 * np.cos(phase * 0.7)
        co_plume = np.exp(-(((x_grid - plume_x) / 0.28) ** 2
                            + ((y_grid - plume_y) / 0.48) ** 2) / 2)
        co_background = 0.12 + 0.035 * np.sin(x_grid * 2 + phase)

        fire_x, fire_y = 0.48 * np.cos(phase * 1.2), 0.42 * np.sin(phase)
        infrared = 0.08 + 0.78 * np.exp(-(((x_grid - fire_x) / 0.12) ** 2
                                           + ((y_grid - fire_y) / 0.12) ** 2) / 2)
        infrared += 0.52 * np.exp(-(((x_grid + 0.35) / 0.16) ** 2
                                    + ((y_grid - 0.25) / 0.14) ** 2) / 2)

        frames[step, :, :, 0] = np.clip(co_background + 0.72 * co_plume
                                        + rng.normal(0, 0.012, (lat, lon)), 0, 1)
        frames[step, :, :, 1] = np.clip(infrared
                                        + rng.normal(0, 0.01, (lat, lon)), 0, 1)

    return frames


def prepare_sequences(data, window_size=10):
    """Turn a (time, lat, lon, channels) array into next-frame sequences."""
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError(
            "data must have shape (time, latitude, longitude, channels); "
            f"received {values.shape}"
        )
    if not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if len(values) <= window_size:
        raise ValueError(
            f"need more than {window_size} time steps, received {len(values)}"
        )

    sequence_count = len(values) - window_size
    X = np.stack([values[index:index + window_size]
                  for index in range(sequence_count)])
    y = np.stack([values[index + window_size]
                  for index in range(sequence_count)])
    return X, y


def load_and_prepare_data(file_path, features=None, target=TARGET,
                          test_size=0.2, random_state=42,
                          group_col=None, log_target=False):
    """Load a fire CSV and return train/test splits plus the fitted preprocessor.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, preprocessor,
                    feature_names, log_target
    """
    features = list(FEATURES if features is None else features)

    logger.info("Loading %s", file_path)
    df = pd.read_csv(file_path)

    # Normalize the official VIIRS FIRMS names used by the bundled dataset.
    aliases = {
        'latitude': 'lat', 'longitude': 'lon',
        'bright_ti4': 'brightness_temp', 'bright_ti5': 'brightness_temp_5',
    }
    df = df.rename(columns={source: target_name for source, target_name in aliases.items()
                            if source in df.columns and target_name not in df.columns})

    # Preserve compatibility with either the original FIRMS names or normalized names.
    features = [aliases.get(c, c) for c in features]

    # --- fail loudly on schema mismatch (issues 1 & 2) ---
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_path} is missing required column(s): {missing}. "
            f"Found: {sorted(df.columns)}"
        )

    # --- coerce to numeric so sentinels like 'NA'/'--' become NaN (issue 5) ---
    for col in features + [target]:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        introduced = df[col].isna().sum() - before
        if introduced:
            logger.warning("%s: %d non-numeric value(s) coerced to NaN",
                           col, introduced)

    # --- drop rows with unknown target, never impute it (issue 4) ---
    n_before = len(df)
    df = df.dropna(subset=[target])
    if len(df) < n_before:
        logger.info("Dropped %d row(s) with missing %s",
                    n_before - len(df), target)
    if df.empty:
        raise ValueError(f"No usable rows in {file_path} after dropping NaN {target}")

    X = df[features].copy()
    y = df[target].copy()

    if log_target:
        y = np.log1p(y.clip(lower=0))

    # --- split before fitting anything (issue 3) ---
    if group_col is not None:
        if group_col not in df.columns:
            raise ValueError(f"group_col {group_col!r} not in {file_path}")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=random_state)
        tr_idx, te_idx = next(splitter.split(X, y, groups=df[group_col]))
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    else:
        logger.warning(
            "Random split on satellite pixels leaks near-duplicates across "
            "train/test; pass group_col (e.g. 'acq_date') for honest metrics."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # impute + scale as one object, fit on train only
    preprocessor = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    logger.info("Split: %d train / %d test, %d features",
                len(X_train), len(X_test), len(features))

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train.to_numpy(),
        'y_test': y_test.to_numpy(),
        'preprocessor': preprocessor,
        'feature_names': features,
        'log_target': log_target,
    }
