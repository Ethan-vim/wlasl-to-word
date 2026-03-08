# Contributing to wlasl-to-word

Thanks for your interest in contributing! This document covers the standards and workflow for making changes to this project.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Workflow](#workflow)
  - [1. Fork and clone](#1-fork-and-clone)
  - [2. Set up the upstream remote](#2-set-up-the-upstream-remote)
  - [3. Create a feature branch](#3-create-a-feature-branch)
  - [4. Make your changes](#4-make-your-changes)
  - [5. Pull from upstream before committing](#5-pull-from-upstream-before-committing)
  - [6. Commit your changes (signed)](#6-commit-your-changes-signed)
  - [7. Push and open a pull request](#7-push-and-open-a-pull-request)
- [Commit Message Format](#commit-message-format)
- [Signed Commits (Required)](#signed-commits-required)
  - [Setting up GPG signing](#setting-up-gpg-signing)
  - [Setting up SSH signing](#setting-up-ssh-signing)
  - [Verifying your setup](#verifying-your-setup)
- [Code Standards](#code-standards)
- [Running Tests](#running-tests)

---

## Prerequisites

- Python 3.10+
- Git 2.34+ (for SSH commit signing) or GPG
- A GitHub account with verified signing keys

---

## Setup

1. **Fork** the repository on GitHub: <https://github.com/Ethan-vim/wlasl-to-word>

2. **Clone** your fork:

   ```bash
   git clone https://github.com/<your-username>/wlasl-to-word.git
   cd wlasl-to-word
   ```

3. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   # .venv\Scripts\Activate.ps1     # Windows PowerShell
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```

5. **Verify tests pass:**

   ```bash
   python -m pytest
   ```

---

## Workflow

### 1. Fork and clone

Fork the repo on GitHub, then clone your fork locally (see [Setup](#setup) above).

### 2. Set up the upstream remote

Add the original repo as `upstream` so you can pull the latest changes:

```bash
git remote add upstream https://github.com/Ethan-vim/wlasl-to-word.git
git remote -v
# origin    https://github.com/<your-username>/wlasl-to-word.git (fetch)
# upstream  https://github.com/Ethan-vim/wlasl-to-word.git (fetch)
```

### 3. Create a feature branch

Always branch off `main`. Never commit directly to `main`.

```bash
git checkout main
git pull upstream main
git checkout -b your-branch-name
```

Use descriptive branch names:

```
feat/webcam-letter-input
fix/mediapipe-import-error
added/download-kaggle-script
docs/update-readme-faq
```

### 4. Make your changes

Write your code, add tests if applicable, and make sure existing tests still pass.

### 5. Pull from upstream before committing

Before committing, pull the latest changes from upstream to avoid merge conflicts:

```bash
git fetch upstream
git rebase upstream/main
```

If there are conflicts, resolve them, then continue:

```bash
# After resolving conflicts in your editor:
git add <resolved-files>
git rebase --continue
```

### 6. Commit your changes (signed)

Stage your changes and create a **signed** commit with a properly formatted message:

```bash
git add <files>
git commit -S -m "feat: add webcam letter input for real-time typing"
```

> **All commits must be signed.** See [Signed Commits (Required)](#signed-commits-required) below for setup instructions. Unsigned commits will not be accepted.

### 7. Push and open a pull request

```bash
git push origin your-branch-name
```

Then open a Pull Request on GitHub against the `main` branch of the upstream repo.

---

## Commit Message Format

All commit messages must follow this format:

```
<type>: <details>
```

Where `<type>` describes the kind of change and `<details>` is a concise description of what was done.

### Types

| Type | When to use | Example |
|------|-------------|---------|
| `feat` | A new feature or capability | `feat: add real-time webcam sign prediction` |
| `fix` | A bug fix | `fix: resolve mediapipe import error on Windows 3.12` |
| `added` | New files, tests, scripts, or assets | `added: kaggle dataset download script` |
| `update` | Enhancement to an existing feature | `update: improve augmentation pipeline with speed perturbation` |
| `refactor` | Code restructuring without behavior change | `refactor: extract keypoint normalization into separate function` |
| `docs` | Documentation only | `docs: add troubleshooting section for CUDA OOM` |
| `test` | Adding or updating tests | `test: add dependency compatibility tests for all libraries` |
| `chore` | Build, CI, config, or tooling changes | `chore: update requirements.txt version ranges` |
| `remove` | Removing code, files, or features | `remove: drop deprecated frame extraction mode` |
| `perf` | Performance improvement | `perf: switch to spawn context for multiprocessing on macOS` |

### Rules

- **Use lowercase** for the type and start details with a lowercase letter.
- **Keep the first line under 72 characters.**
- **Use the imperative mood** in details: "add webcam support" not "added webcam support" or "adds webcam support".
- **Do not end with a period.**
- If more context is needed, add a blank line then a longer body:

  ```
  fix: resolve zero-division in shoulder normalization

  When all frames have zero shoulder width (detection failure), the
  normalization would crash. Now falls back to scale=1.0 when no
  valid widths are found.
  ```

### Examples

```
feat: add ONNX export with dynamic batch size
fix: handle empty video files in keypoint extraction
added: 110 dependency compatibility tests
update: support WLASL300 and WLASL1000 variants
refactor: consolidate model building into single factory
docs: document auto-config hardware detection
test: add confusion matrix shape assertions
chore: pin mediapipe to <=0.10.14 for stability
remove: unused decord fallback in video dataset
perf: enable persistent workers in DataLoader
```

---

## Signed Commits (Required)

**Every commit must be cryptographically signed.** This verifies that commits genuinely come from the listed author. GitHub will show a "Verified" badge next to signed commits.

Unsigned or unverified commits will not be merged.

### Setting up GPG signing

1. **Generate a GPG key** (if you don't have one):

   ```bash
   gpg --full-generate-key
   # Choose: RSA and RSA, 4096 bits, key does not expire
   # Use the same email as your GitHub account
   ```

2. **Get your key ID:**

   ```bash
   gpg --list-secret-keys --keyid-format=long
   # Look for the line: sec   rsa4096/XXXXXXXXXXXXXXXX
   # XXXXXXXXXXXXXXXX is your key ID
   ```

3. **Add the key to GitHub:**

   ```bash
   gpg --armor --export XXXXXXXXXXXXXXXX
   ```

   Copy the output and add it at: **GitHub > Settings > SSH and GPG keys > New GPG key**

4. **Configure git to use your key:**

   ```bash
   git config --global user.signingkey XXXXXXXXXXXXXXXX
   git config --global commit.gpgsign true
   ```

   With `commit.gpgsign true`, all commits will be signed automatically (no need for `-S` flag).

### Setting up SSH signing

Git 2.34+ supports signing with SSH keys, which is simpler if you already use SSH for GitHub:

1. **Configure git to use SSH signing:**

   ```bash
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/id_ed25519.pub
   git config --global commit.gpgsign true
   ```

2. **Add your SSH key as a signing key on GitHub:**

   Go to **GitHub > Settings > SSH and GPG keys > New SSH key**, set type to **Signing Key**, and paste your public key.

### Verifying your setup

After configuring, test that signing works:

```bash
echo "test" | git commit --allow-empty -S -m "test: verify commit signing"
git log --show-signature -1
```

You should see `Good signature` in the output. Delete the test commit afterward:

```bash
git reset HEAD~1
```

---

## Code Standards

- Follow existing code style and patterns in the codebase.
- Add tests for new features. Run `python -m pytest` before committing.
- Do not commit secrets, credentials, `.env` files, or large binary files.
- Keep pull requests focused: one feature or fix per PR.
- Update `README.md` and `STRUCTURE.md` when adding features, scripts, or changing project structure.

---

## Running Tests

```bash
python -m pytest                          # full suite (277 tests)
python -m pytest tests/test_augment.py    # single file
python -m pytest tests/test_dependencies.py  # dependency compatibility
python -m pytest -q                       # quiet output
python -m pytest -x                       # stop on first failure
```

All tests must pass before a PR will be reviewed.
