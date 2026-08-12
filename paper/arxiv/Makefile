PYTHON ?= ../../.venv/bin/python
LATEXMK ?= latexmk

.PHONY: all generated clean distclean

all: main.pdf

generated:
	$(PYTHON) ../../scripts/build_paper_artifacts.py

main.pdf: generated main.tex references.bib $(wildcard sections/*.tex)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex

distclean: clean
	rm -f generated/results-macros.tex generated/trace-primary-table.tex \
		generated/resident-primary-table.tex generated/paper-data-manifest.json
