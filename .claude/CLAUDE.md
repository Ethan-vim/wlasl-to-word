/age# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Always follow instructions in CLAUDE.md (this file#)
## Project Overview

Live American Sign Language (ASL) recognition system using the WLASL (Word-Level American Sign Language) dataset. This project is in early development — no source code exists yet.

## Environment

- Python 3.12 (managed via `../.venv`)
- Activate: `source ../.venv/bin/activate`
- Install dependencies (once requirements.txt is created): `pip install -r requirements.txt`
- Read PROJECT_DES.md
- Read README.md
- Run all unit tests to ensure reliability, if unit tests fail, fix issues.

## Codebase Adaptation/Understanding
- Read PROJECT_DES.md
- Read README.md
- Always ask user if something is unclear
- Read all files in the project folder

# File edits

Clarity when editing files.

## Adjusting Codebase
A lot of times, after making new features, important things to the codebase, they won't adapt into the rest of the codebase.
So there often needs other file changes

For example,

After implementing a feature that allows a user to contact customer service.
The frontend might not be linked to that feature, resulting in inefficient usage.

In a repository, the README.md might not be up-to-date with the new files after every new edit, feature, etc.
Those edits related to user usage, setup, installation, methods, must all be added into README.md to clarify the features added.

For example,

A new feature has been implemented in the codebase that allows the user to use a webcam to deliver WLASL to the transformer model.
README.md isn't frequently updated, so that feature is not written into README.md.

That feature must be written into README.md to make know how to use that feature.

Overall, all file edits should be adapted into the codebase.  
I also state explicitly that new features must be written into README.md with a usage-guide and other info about it.

## Conclusion

After every user prompt, at the end of the output, state what has been changed in each file, what has been added, what is fixed, etc.

Each file edit prompt output should follow this structure:

```
<File_Name>:
        - Bug Fixes: 
        - Feature implementations:
        - Usage:

<File_Name>:
        - Bug Fixes: 
        - Feature implementations:
        - Usage:

Conclusion: # Generic
        - Features implementations:
        - Bug fixes: 
        - Usage:
```

Each explanation should be concise, clear, and explanatory.

# Python

General stuff for Python Programming language

## Machine/Deep Learning
        
When making a Machine/Deep learning project with python, the folder structure should look something like this:
<hash_tag> = comment

```
checkpoints/
        best_model.pt
configs/
data/
        annotations/
        processed/
        raw/
        splits/
logs/
notebooks/
scripts/
src/ # Python Package(s) including itself
        __init__.py
        data/
                __init__.py
        inference/
                __init__.py
        models/
                __init__.py
        training/
                __init__.py
tests/ # Python Package
        __init__.py
requirements.txt
README.md
```

Other files may be added to ensure git tracking, functionality.  This is just a folder/file structure.