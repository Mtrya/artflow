---
name: distill-agent-rule
description: Manually distill a durable agent-behavior rule from a user complaint about something an agent did — interrogate the incident against its concrete evidence one point at a time, formulate the general prohibition with its temptation, place it in AGENTS.md, a skill, or a mechanical gate, and audit the existing rule corpus for supersession, merge, and deletion. Runs only on explicit user invocation; an agent must never distill a rule about its own behavior unprompted.
disable-model-invocation: true
user-invocable: true
---

# Distill an agent rule

Turn one user complaint about agent behavior into a durable, general rule with teeth. The complaint is the input; the rule is the product. Guidance, not a script.

## Invocation boundary

Run only when the user explicitly invokes `distill-agent-rule` by name. A rule written by the perpetrator, from its own vantage, is self-report, not evidence.

## Fix the evidence first

Get the concrete artifact before any formulation: the diff, the test file, the command output, the job log. The complaint is a vantage, not ground truth. No artifact, no rule — ask for it or find it, and stop if it cannot be found.

## Interrogate one point at a time

Ask one question, get one answer, dig until the point is exhausted, then move on. Required findings:

- what the agent did;
- what it should have done;
- **why the wrong move looked reasonable at that moment** (the temptation);
- what cheap observation would have exposed it on the spot.

## Formulate the class, not the incident

Rule template:

> **Never X** (temptation: why X looks reasonable in the moment). **Do Y instead** (the criterion that decides).

The temptation sentence is mandatory: the future agent meets the temptation, not the incident; without it, it cannot recognize the rule as its own.

When the user states a philosophy during interrogation, the rule carries that principle as its core — do not re-derive what the user already said better.

Two controls before a rule may land:

- **Positive control:** had the rule been loaded at the time, would it have changed the action in this incident? If not, it is a platitude — sharpen or drop it.
- **Overreach check:** does it forbid anything legitimate? Name the exception inside the rule. If exceptions keep multiplying, the class is drawn wrong.

## Search before writing

Read `AGENTS.md`, `.agents/skills/`, and `~/.agents/skills/` before drafting. Every new rule triggers a scoped audit of existing entries covering the same behavior:

- an existing rule covers it → sharpen the existing rule, do not add;
- the new rule absorbs every unique proposition of an old one → merge and delete the old;
- a rule in scope is no longer load-bearing → delete it in the same pass;
- never defer a known overlap to a later audit; never work toward a quota.

## Place the rule

- Universal to every task and statable in a line or two → `AGENTS.md`. That file stays ruthlessly short: every line must change an action, so a new rule may require merging or evicting an old one.
- A domain workflow with judgment procedure and calibration examples → a skill under `.agents/skills/`.
- Mechanically checkable → prefer a gate (a test or lint check) over prose; the prose rule then only names the gate.

## Keep the incident as calibration

The rule text must pass the HEAD test: a reader with no access to this conversation can resolve every reference. The incident itself goes into the owning skill's examples file as a dated calibration example — evidence, not authority.

## Approval and report

Draft rules are proposals; the user approves each rule and its placement before anything is written. Report: rules added / updated / merged / deleted, placements, gates created, and the artifacts used as evidence.
