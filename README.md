# Derivative Term-sheet Analyser

Python research prototype developed by Team C for the UBS Tomorrow Talent Program. This repository organises the supplied team notebook into source files, a credential-free offline example and an experimental full workflow.

**Team project, not a UBS product or an endorsed production system.** Team members in the supplied deck: CHEN Hongyu, LIU Haolin, PING An, TAN Xin Wei and WANG Ruotong (Elleny). Module ownership has not been independently verified; inclusion here does not imply sole authorship by Elleny.

## What it does

The research pipeline extracts PDF text, uses an OpenAI-compatible LLM to structure key terms, separates coupon/knock-in/autocall barriers, analyses product logic and renders payoff explanations and illustrative scenarios. The original notebook also contains experimental pricing code; its presence is not evidence of validated valuation accuracy or a working production pricing engine.

```text
PDF → text → parameter extraction → product logic → visual explanations
```

## Quick start — offline, no API key

Requires Python 3.11+. No third-party libraries are needed for this example.

```sh
python cli.py examples/synthetic_termsheet.txt
python -m unittest discover -s tests -v
```

The offline command runs the original structural-tag function on synthetic text. It does not perform LLM extraction or parse PDFs. Expected tags include memory, step-down, worst-of/autocall flags and observation frequency; the supplied example sets memory, step-down and autocall to true, quarterly frequency and worst-of to false.

## Full experimental workflow

```sh
python -m venv .venv
# Activate the environment using your platform's normal command.
python -m pip install -r requirements.txt
```

Install the Graphviz system executable separately if flowcharts are required. Set these environment variables in your shell: `TERMSHEET_API_KEY`, `TERMSHEET_BASE_URL`, `TERMSHEET_MODEL`. The example file lists names only; it is not loaded automatically. Use a model and endpoint available to your account; defaults are inherited from the project context, not a promise of current provider availability.

```sh
python run_pipeline.py private_inputs/example.pdf --output outputs
```

This runner captures notebook HTML/text display outputs. Full API/PDF/Graphviz execution has **not** been validated as part of the repository packaging. The original research code retains assumptions and error-handling behaviour from the notebook. Review logs and generated content carefully. Use the notebook for the original interactive display experience.

## Files

| File | Purpose |
| --- | --- |
| `termsheet_core.py` | Original, pure extraction helpers |
| `cli.py` | Reproducible offline tag demo |
| `research_pipeline.py` | Sanitised original research functions and prompts |
| `run_pipeline.py` | New experimental PDF runner and display capture |
| `notebooks/research.ipynb` | Clean notebook, no stored output or credentials |
| `examples/` | Synthetic input only |
| `tests/` | Offline behavioural checks |

## Validation and limitations

Offline tag behaviour and Python syntax are tested. LLM quality, PDF variation, generated formulas, flowcharts, path monitoring conventions, pricing and numerical convergence are not validated here. Structural tags are substring/regex hints and can misclassify negated or ambiguous text. Human review against the original term sheet is required. Scenario paths are educational illustrations, not calibrated market forecasts. No performance or valuation-accuracy claims are made.

## Credentials and disclosure

Keys embedded in the original source were excluded. Notebook outputs, metadata and uploaded documents are not copied. Rotate any formerly embedded keys before using this project. API input is sent to the configured model provider; use synthetic or authorised documents only. Keep secrets in environment variables and private inputs outside version control. `.gitignore` reduces accidental inclusion but is not a secret scanner. Inspect every commit before sharing.

## Provenance and reuse

Derived from the user-supplied Team C notebook and presentation. Packaging, environment configuration, offline CLI, tests and runner were added for this portfolio. ASTRI material is not part of this repository. No open-source licence is asserted on behalf of other contributors; agree on reuse and publication rights with the team before granting third parties a licence.
