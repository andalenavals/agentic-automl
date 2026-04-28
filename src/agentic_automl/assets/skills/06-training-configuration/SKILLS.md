---
name: automl-06-training-configuration
description: Use when executing the training step of Agentic AutoML. Choose the concrete training parameters for the selected model while preserving reproducibility.
---

# Training Configuration Skill

## Goal

Set a reproducible training configuration that matches the selected model and the dataset size.

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
- accepted earlier workflow decisions

## Procedure

1. Default to reproducible settings with a fixed random seed.
2. Expose only the training parameters that are relevant for the selected model.
3. Include optimizer, learning rate, epochs, and mini-batch size when the model supports them.
4. Include cross-validation or validation-budget settings used during training.
5. Explain which parameters are active versus not applicable for the current model.

## Output

- selected training policy
- explicit training parameters
- short justification tied to reproducibility and model fit
