# Quick Reference: Git Commands

## Branching

`git checkout -b feature-x` creates and switches to a new branch. `git branch -d feature-x` deletes it once merged; use `-D` to force-delete an unmerged branch.

## Rebasing

`git rebase main` replays your commits on top of the latest main. Use `git rebase -i HEAD~5` to interactively squash or reword the last 5 commits before pushing.

## Stashing

`git stash` shelves uncommitted changes. `git stash pop` reapplies the most recent stash and removes it from the stash list; `git stash apply` reapplies without removing it.

## Tags

`git tag v1.2.0` creates a lightweight tag. `git tag -a v1.2.0 -m "message"` creates an annotated tag with metadata — prefer annotated tags for releases.

## Submodules

`git submodule add <url> path/to/submodule` adds one. After cloning a repo with submodules, run `git submodule update --init --recursive` to actually pull their contents.
