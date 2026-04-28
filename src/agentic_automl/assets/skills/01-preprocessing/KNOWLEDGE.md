# Preprocessing Knowledge

## Current Capability

This step can inspect the dataset file, recommend a base preprocessing policy, and build executable custom preprocessing overrides for common tabular requests.

## Supported Base Policies

- `minimal_cleanup`
- `auto_tabular_preprocessing`
- `custom`

## Capability Keys

- `scope_keep_subset`: keep only a named subset of features
- `scope_drop_named_features`: drop one or more named features
- `scope_keep_named_features`: keep one or more named features
- `scope_only_numeric`: use only numeric features
- `scope_only_categorical`: use only categorical features
- `scope_only_date_like`: use only date-like features
- `scope_exclude_numeric`: exclude numeric feature families
- `scope_exclude_categorical`: exclude categorical feature families
- `scope_exclude_date_like`: exclude date-like feature families
- `role_as_categorical`: treat a feature as categorical
- `role_as_numeric`: treat a feature as numeric
- `role_as_date_like`: treat a feature as date-like
- `role_as_identifier`: treat a feature as an identifier
- `encoding_one_hot`: one-hot encode a named feature
- `encoding_ordinal`: ordinal encode a named feature
- `impute_mean`: apply mean imputation to a numeric feature
- `impute_median`: apply median imputation to a numeric feature
- `impute_most_frequent`: apply most-frequent imputation to a feature
- `impute_constant`: apply constant-value imputation to a feature
- `scale_standard`: scale a numeric feature with `StandardScaler`
- `scale_minmax`: scale a numeric feature with `MinMaxScaler`
- `scale_robust`: scale a numeric feature with `RobustScaler`
- `scale_none`: keep a numeric feature unscaled
- `transform_log`: apply a log transform to a numeric feature
- `transform_log1p`: apply a log1p transform to a numeric feature
- `transform_sqrt`: apply a square-root transform to a numeric feature
- `transform_square`: square a numeric feature
- `transform_abs`: apply absolute value to a numeric feature
- `transform_multiply_constant`: multiply a numeric feature by a constant
- `transform_divide_constant`: divide a numeric feature by a constant
- `transform_add_constant`: add a constant to a numeric feature
- `transform_subtract_constant`: subtract a constant from a numeric feature
- `derive_frequency_count`: create a frequency-count column from a source feature

## Supported Action-Mode Changes

### Feature Scope

- keep only a named subset of features
- drop one or more named features
- keep one or more named features
- use only numeric features
- use only categorical features
- use only date-like features
- exclude numeric, categorical, or date-like feature families

### Feature Role Overrides

- treat a feature as categorical
- treat a feature as numeric
- treat a feature as date-like
- treat a feature as an identifier

### Encoding

- one-hot encode a named feature
- ordinal encode a named feature

### Imputation

Supported per-feature imputation rules:

- `mean`
- `median`
- `most_frequent`
- constant-value imputation with an explicit value such as:
  - `0`
  - `-1`
  - `999`
  - `'unknown'`

Notes:

- `mean` and `median` are only supported for numeric features
- out-of-range imputation is supported when the replacement value is explicit

### Scaling

Supported per-feature scaling rules for numeric features:

- `standard` via `StandardScaler`
- `minmax` via `MinMaxScaler`
- `robust` via `RobustScaler`
- `none`

### Numeric Transformations

Supported per-feature numeric transformations:

- `log`
- `log1p`
- `sqrt`
- `square`
- `abs`
- multiply by a constant
- divide by a constant
- add a constant
- subtract a constant

### Derived Features

Supported derived feature actions:

- create a frequency-count column from a source feature

Current output naming rule:

- `<feature>__frequency_count`

## Runtime Behavior

- custom requests are merged incrementally across Action-chat turns
- the executed preprocessing graph follows the merged custom policy
- supported custom actions appear in the generated execution steps
- unsupported Action-mode requests are not executed, do not overwrite the current working policy, and are appended to `LIMITS.md`
