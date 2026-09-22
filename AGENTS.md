# doublewordai/frontend-crates fork layout

Integration fork of ai-dynamo/frontend-crates. The doublewordai/dynamo fork
consumes these crates from crates.io and points the crates it patches at
this repo's `main` with a cargo `[patch]`.

## Branches

- `upstream-base`: exactly the upstream `main` commit the fork is based on.
  Currently `bb20dd01b625` (2026-09-22: dynamo-parsers 9.0.1,
  dynamo-parsers-v2 0.6.3, dynamo-renderer 5.3.2). Never carries our commits.
- `main`: `upstream-base` plus every patch branch below, merged with
  `--no-ff` in stack order. `git log --merges upstream-base..main` is the
  patch list.
- `upstream-pr/<topic>`: a change we intend to land upstream. Based on
  `upstream-base`. This repo is a public fork, so the branch heads the
  upstream PR directly once rebased onto upstream `main`.
- `vendor/<topic>`: a Doubleword-only change that will not go upstream.
  Based on `upstream-base`.
- `archive/*` and other branches are history.

## Rules

- No backports. Do not cherry-pick upstream commits onto `main`; a fix that
  is on upstream `main` arrives by moving `upstream-base`.
- One branch per patch, atomic, with the reason in the commit message.
- Moving the base: point `upstream-base` at the new upstream `main` commit,
  rebase each patch branch that is still needed onto it, drop the ones
  upstream now contains, rebuild `main` as base plus merges, force-push
  `main`. Update the stack list below.
- A patched crate keeps the version upstream published at the base, so the
  dynamo fork's `[patch]` stays semver-compatible with its pin.

## Current stack

- `vendor/fork-layout`: this section.
- `upstream-pr/glm47-v1-no-entity-decode`: deliver GLM tool-call argument
  values as written; no XML entity decoding.
- `upstream-pr/glm47-array-no-split`: keep GLM tool-call array arguments
  whole; no comma splitting.

## Planned stack (agreed 2026-09-22)

- `upstream-pr/hunyuan-parser`: Hunyuan reasoning and tool-call parsing,
  moved here from the dynamo fork.
- `upstream-pr/mimo-parser`: MiMo reasoning and tool-call parsing, moved
  here from the dynamo fork.

---

# frontend-crates agent instructions

- Before v2 conformance work, read `conformance/README.md`, "V2 storage contract: plain versioned YAML only". #257 removes legacy formats and redundant capture identities; old readers are migration work, not permission to add archives, patch files, or hash-qualified names.
- After any frontend-crates conformance fixture, parser, table-model, renderer, or conformance-documentation change, render the canonical report with `conformance/utils/render_table_v2.sh`, which writes `conformance/CONFORMANCE_v2.html`. Do this before reporting completion; do not stage or commit the generated HTML unless explicitly requested. After every render, report both its absolute filesystem path and its served `http://keivenc-linux1/dev/<worktree>/conformance/CONFORMANCE_v2.html` URL.
- A request to port, convert, or add a model family to `UnifiedParser` means a full conversion. `UnifiedParser` and its shared scanner must become the only parser state machine for that family. Remove the old bespoke v2 parser state machine; retain a legacy `ToolParser` entry point only when external callers still require it, and make it a thin projection of the same `UnifiedParser` events. Do not report a factory or registry entry as a conversion while a second family-specific buffer, scanner, drain loop, or grammar owner remains.
- A UnifiedParser conversion is incomplete until its current crate-version capture has been collected: generate the authored Unified inputs and golden events, run the live Unified capture, publish plain-version YAML and required manifest changes, extract the snapshot, and render `conformance/CONFORMANCE_v2.html`. Do not generate `CONFORMANCE_unified.html`.
- Before reporting a UnifiedParser conversion complete, run `conformance/utils/check.sh status --model <family> --tab unified`. It must report zero red and zero empty current Dynamo cells. Also run the native parser tests, Unified golden parity, the live-vs-packaged-capture guard, and the v2 table-model tests. A generated loose feed, a parser unit test, or a red diagnostic capture is not completion.
