# Data Splitting Knowledge

## Current Capability

This step can choose and execute one of the packaged final-holdout strategies.

## Supported Base Policies

- `stratified_holdout`
- `random_holdout`
- `time_ordered_holdout`

## Supported Action-Mode Changes

- choose a stratified holdout for classification
- choose a random holdout
- choose a time-ordered holdout when chronology matters
- discuss leakage implications of the chosen split

## Capability Keys

- `split_stratified_holdout`: choose a stratified holdout for classification
- `split_random_holdout`: choose a random holdout
- `split_time_ordered_holdout`: choose a time-ordered holdout when chronology matters
- `split_explain_leakage`: discuss leakage implications of the chosen split
