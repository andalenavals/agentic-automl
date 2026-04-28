Action Agents
=============

Shared philosophy
-----------------

All Action-mode step agents follow the same contract:

1. Natural follow-up conversation
2. Awareness of previous context and policy selections
3. Awareness of limits and knowledge
4. Clear distinction between actions and consultation
5. Suggestions based on questions, policies, and current customizations
6. Flexibility to revise policy multiple times inside one conversation

Runtime behavior
----------------

The shared coordinator lives in ``src/agentic_automl/ui_logic.py``.

When the human asks a question in Action mode:

* the answer is treated as consultation rather than execution
* the active policy remains unchanged
* the reply shows the current action context
* the reply suggests useful next supported moves

When the human asks for an executable change:

* the request is parsed into step-local intents
* intents are mapped to capability keys from ``KNOWLEDGE.md``
* supported intents mutate the working policy
* unsupported intents leave the policy unchanged and are written to
  ``LIMITS.md``

Step-specific policy modules
----------------------------

The shared coordinator delegates execution parsing to step modules:

* ``preprocessing_actions.py``
* ``split_actions.py``
* ``model_actions.py``
* ``metric_actions.py``
* ``training_actions.py``
* ``hpo_actions.py``
* ``validation_actions.py``
* ``final_validation_actions.py``

These modules are responsible for step-local intent parsing and for keeping the
runtime honest about what can really execute.

Multi-turn revisions
--------------------

Action mode is designed to support sequential revisions. A user can:

* choose a policy
* refine it later
* replace it later again
* ask consultation questions in between

without losing the accepted context from earlier turns.
