# Repository Guidelines

## Project Structure & Module Organization
- `main.tex`: paper entrypoint (ICML 2026 style via `icml2026.sty`).
- `Section/*.tex`: main paper sections (e.g., `Section/Introduction.tex`, `Section/Approach.tex`).
- `Section/Appendix.tex` + `Section/AppendixParts/*.tex`: appendix roadmap, algorithms, derivations, proofs, and extra experiment details.
- `biblio.bib`: BibTeX database; `icml2026.bst`: bibliography style.
- `figures/`: PDF figures referenced by the paper.
- `old/`, `out/`: scratch/outputs (avoid editing unless necessary).

## Build, Test, and Development Commands
This repo is LaTeX-only (no unit tests). The main “check” is successful compilation with no undefined references.
- Build PDF: `latexmk -pdf -interaction=nonstopmode -file-line-error main.tex`
- Clean aux files: `latexmk -c` (or `latexmk -C` to remove the PDF too)

## Writing Style & Naming Conventions
- Prefer concise ICML tone: precise, restrained, and technical.
- Keep notation consistent across sections; introduce symbols before first use.
- Use `\label{}` + `\ref{}`/`\eqref{}` consistently; labels should follow a clear prefix pattern:
  - Sections: `sec:...`, equations: `eq:...`, figures: `fig:...`, tables: `tab:...`, appendices: `app:...`.
- Avoid overly long sentences; break dense paragraphs into 2–4 sentence blocks.

## Figures & Bibliography
- Store figures in `figures/` and reference via `\includegraphics{figures/...}`.
- Add citations to `biblio.bib`; keep BibTeX keys stable once used.

## Pre-PR Checklist
- Compile `main.tex` and confirm `main.log` has no “undefined references” warnings.
- Ensure new/changed labels are unique and referenced correctly.

## Commit & Pull Request Guidelines
- Git history may not be present in this workspace; if available, match existing commit style.
- Recommended commit format: short imperative subject (e.g., “Fix notation in problem formulation”), optional body explaining rationale.
- PRs should include: a brief summary of changes, affected section paths, and compilation status (command + result).
