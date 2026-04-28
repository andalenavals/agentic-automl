---
name: automl-01-preprocessing
description: Use when executing the preprocessing step of Agentic AutoML. Inspect the real dataset file, identify preprocessing risks, and choose an executable preprocessing policy that matches the runtime pipeline.
---

# Preprocessing Skill

## Goal

Choose a preprocessing policy that is justified by the actual dataset profile and that stays aligned with the executed pipeline.

This step now also owns feature pruning, feature subsetting, and feature-role changes. There is no separate workflow step for feature selection anymore.

## Action Agent Philosophy

- Keep Action mode conversational across natural follow-up turns.
- Use the accepted brief and previous workflow choices as context.
- Distinguish consultation from execution: questions explain, requests mutate.
- Stay inside `KNOWLEDGE.md`, and keep policy unchanged when a request falls outside the executable space.
- Store unsupported requests in `LIMITS.md` and suggest the next supported move when possible.
- Allow the human to revise the active policy multiple times across the same conversation without losing the accepted context.

## Inputs

- `ProjectBrief`
- `DataProfile`
- current dataset file inspection signals

## Focus Areas

- missingness by column
- numeric versus categorical features
- whether a feature's semantic role should be overridden from the raw dtype
- constant columns
- likely identifier columns
- date-like columns
- low-cardinality categorical columns
- high-cardinality categorical columns
- sparse features or columns that are mostly empty

## Cardinality Reasoning

- Treat low-cardinality categorical features as the normal one-hot encoding case.
- Treat high-cardinality categorical features as a preprocessing risk that can explode width, overfit, or create brittle rare-category behavior.
- If categorical cardinality is low and the rest of the file looks healthy, prefer `minimal_cleanup` or `auto_tabular_preprocessing` depending on missingness and other risks.
- If categorical cardinality is high, prefer `custom` and recommend one of these procedures:
  - drop obvious identifier-like categoricals
  - keep only the most meaningful high-cardinality fields after review
  - avoid blind one-hot expansion when it would create an excessively wide sparse matrix
- In the reasoning, name the affected columns and explain whether the issue is width, rarity, leakage risk, or likely identifier behavior.

## Sparsity Reasoning

- Look for sparse signals such as columns that are mostly missing, very rare categories, or feature spaces that would become extremely wide and mostly zero after encoding.
- If sparsity is mild and localized, `auto_tabular_preprocessing` can still be acceptable with targeted imputation.
- If sparsity is severe, prefer `custom` and recommend one or more of these procedures:
  - drop columns that are mostly empty and unlikely to carry stable signal
  - keep imputation targeted instead of applying broad transformations everywhere
  - avoid unnecessary scaling or expansion for features that are mostly absent
  - review whether the sparse feature should be represented differently before modeling
- Explain whether the main concern is information quality, matrix width, instability from rare values, or wasted model capacity.

## Procedure

1. Profile the real dataset file before recommending anything.
2. Reason explicitly about missingness, date parsing failures, cardinality, and sparsity before choosing the policy.
3. Check whether any feature should be reclassified semantically, for example treating an integer-coded feature as categorical, a string column as date-like, or a column as an identifier instead of predictive input.
4. When a feature-role override is needed, switch to `custom` and make the override explicit in the execution plan.
5. Also understand set-level scope requests such as using only numeric features, only categorical features, or excluding an entire feature family, and convert those requests into explicit executable preprocessing changes.
6. Understand feature-level encoding requests such as one-hot encoding, ordinal encoding, or treating a numeric-coded feature as categorical, and convert those requests into explicit runtime overrides.
7. Understand explicit feature-subset allowlists such as `use only age and tenure_months`, `keep only these columns`, or `select only these features`, and convert them into an executable subset plan that removes every other input feature.
8. Treat Action-mode requests as incremental edits to the current working preprocessing plan. Do not recompute from scratch and forget earlier accepted customizations unless the new instruction clearly replaces them.
9. If a new custom request conflicts with an earlier one, reconcile the working plan and explain the merged result that will actually run.
10. Recommend numeric imputation only when the inspected file actually shows missing numeric values or when date parsing will create missing date-derived values.
11. Recommend categorical imputation only when the inspected file actually shows missing categorical values.
12. If numeric scaling is recommended, name the scaling method explicitly and explain why that method is the default for the current candidate set.
13. Prefer `minimal_cleanup` when the file is already clean and does not show special risks.
14. Prefer `auto_tabular_preprocessing` for standard mixed tabular cases with ordinary missingness and manageable categorical width.
15. Prefer `custom` when the file shows likely IDs, date-like fields, constant columns, risky categorical cardinality, severe sparsity, or explicit feature-role, feature-encoding, or feature-subset overrides from the user.
16. A feature-specific transformation request that changes how a column moves through the preprocessing graph must resolve to `custom`, not back to the generic automatic policy.
17. For every execution step, cite the concrete checked feature facts that triggered it, such as missing fractions, distinct counts, row uniqueness, date parse failures, or numeric ranges.
18. Explain the decision with file-specific reasons, not generic ML advice.
19. Keep the recommendation aligned with the executable preprocessing policy used by the training pipeline.
20. In Action mode, first convert the request into step-local executable intents, then map those intents to capability keys from `KNOWLEDGE.md`.
21. If any requested capability key is missing from `KNOWLEDGE.md`, do not mutate the current working policy. Explain that the request is outside the current executable space and append it to `LIMITS.md`.
22. Unsupported follow-up requests must never erase an already valid custom preprocessing policy.

## Output

- selected preprocessing option
- concise reasoning bullets tied to the inspected file
- explicit execution steps only for the transforms that will actually run
- explicit feature-role override steps when the user changes how a column should be treated
- explicit feature-encoding override steps when the user asks for a concrete encoding method on one or more columns
- explicit feature-subset steps when the user wants to keep only a named set of input features
- explicit scope-override steps when the user wants only a subset of feature families such as numeric-only or categorical-only preprocessing
- never show categorical imputation as a step unless categorical gaps are present in the inspected file
- one argumentation line for each execution step, including the checked feature facts behind that step and the exact scaling method when scaling is used
- when a step is customized through multiple Action-chat turns, show the merged working plan rather than only the most recent instruction
- explicit question asking whether the human agrees or wants something different
