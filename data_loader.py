import logging
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURES = ['lat', 'lon', 'brightness_temp', 'precip_mm_day', 'soil_moisture']
TARGET = 'frp'


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
