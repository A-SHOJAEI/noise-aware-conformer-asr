PIP ?= .venv/bin/pip
.PHONY: setup data train eval report all clean

PYTHON ?= .venv/bin/python
VENV := .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

CONFIG ?= configs/smoke.yaml
RUN_DIR ?= runs/smoke
ARTIFACTS_DIR ?= artifacts

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@./scripts/bootstrap_venv.sh "$(PYTHON)" "$(VENV)"
	@$(VENV_PIP) install -r requirements.txt

data:
	@$(VENV_PY) -m asr.data.smoke --out data/smoke

train:
	@$(VENV_PY) -m asr.train --config "$(CONFIG)" --run-dir "$(RUN_DIR)" --overwrite

eval:
	@mkdir -p "$(ARTIFACTS_DIR)"
	@$(VENV_PY) -m asr.eval --run-dir "$(RUN_DIR)" --out "$(ARTIFACTS_DIR)/results.json"

report:
	@mkdir -p "$(ARTIFACTS_DIR)"
	@$(VENV_PY) -m asr.report --results "$(ARTIFACTS_DIR)/results.json" --out "$(ARTIFACTS_DIR)/report.md"

all: setup data train eval report

clean:
	@rm -rf "$(VENV)" runs artifacts __pycache__

