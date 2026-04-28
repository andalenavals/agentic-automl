# Intake Knowledge

## Current Capability

This step can capture and validate the project brief that the rest of the workflow depends on.

## Supported Actions

- accept the brief gradually across multiple messages
- recognize aliases such as `dataset`, `target`, `task`, and `problem`
- infer `project_name` from the dataset filename when it is missing
- validate that `task_type` is either `classification` or `regression`
- keep optional fields when provided:
  - `date_column`
  - `baseline_metric`
  - `competition_enabled`

## Action Chat Scope

- restate the current brief
- correct or replace any captured intake field
- add missing optional fields

## Capability Keys

- `intake_accept_gradual_brief`: accept the brief gradually across multiple messages
- `intake_alias_fields`: recognize field aliases such as `dataset`, `target`, `task`, and `problem`
- `intake_infer_project_name`: infer `project_name` from the dataset filename when it is missing
- `intake_validate_task_type`: validate that `task_type` is `classification` or `regression`
- `intake_set_date_column`: capture or replace the optional `date_column`
- `intake_set_baseline_metric`: capture or replace the optional `baseline_metric`
- `intake_set_competition_enabled`: capture or replace the optional `competition_enabled`
- `intake_restate_brief`: restate the current captured brief
- `intake_update_field`: correct or replace any captured intake field
