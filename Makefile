SHELL := /bin/bash

VENV = venv
VENV_BIN = $(VENV)/bin
PIP := $(VENV_BIN)/pip

deps: $(VENV_BIN)/activate
	$(PIP) install -r requirements.txt

$(VENV_BIN)/activate:
	virtualenv $(VENV)

