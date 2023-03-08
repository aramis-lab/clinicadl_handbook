POETRY ?= poetry
CONDA ?= conda
CONDA_ENV ?= "./env"
PYTHON_SCRIPTS_DIR := src
NOTEBOOKS_DIR := notebooks
BOOK_DIR := jupyter-book
MINIMAL_NOTEBOOK_FILES = $(shell ls $(PYTHON_SCRIPTS_DIR)/*.py | perl -pe "s@$(PYTHON_SCRIPTS_DIR)@$(NOTEBOOKS_DIR)@" | perl -pe "s@\.py@.ipynb@")

.PHONY: help
help: Makefile
	@echo "Commands to use ClinicaDL Makefile:"
	@sed -n 's/^##//p' $<

## env             : Bootstrap an environment
.PHONY: env
env: env.dev

## env.conda       : Create a Conda environment
.PHONY: env.conda
env.conda:
	@$(CONDA) env create -f environment.yml -p $(CONDA_ENV)

## env.dev         : Install with Poetry
.PHONY: env.dev
env.dev:
	@$(POETRY) install

## build.notebooks : Build notebooks from source using jupytext
.PHONY: build.notebooks
build.notebooks: $(NOTEBOOKS_DIR)

$(NOTEBOOKS_DIR): $(MINIMAL_NOTEBOOK_FILES)

$(NOTEBOOKS_DIR)/%.ipynb: $(PYTHON_SCRIPTS_DIR)/%.py
	@$(POETRY) run jupytext --to notebook $< --output $@

## clean.book      : Clean jupyter-book (except for cache)
.PHONY: clean.book
clean.book:
	@$(POETRY) run jupyter-book clean $(BOOK_DIR)

## full-clean.book : Full clean of jupyter-bool
.PHONY: full-clean.book
full-clean.book:	
	@$(RM) -rf $(BOOK_DIR)/_build

## build.book      : Build the jupyter-book
.PHONY: build.book
build.book: clean.book
	@$(POETRY) run jupyter-book build $(BOOK_DIR)

## sanity-check    : Verify if all py files have a correspondent notebook (ipynb file)
.PHONY: sanity-check
sanity-check:
	python .build/sanity-check.py $(PYTHON_SCRIPTS_DIR) $(NOTEBOOKS_DIR)
