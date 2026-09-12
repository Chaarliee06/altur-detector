PYTHON ?= python3
PY := .venv/bin/python
WORKERS ?= 6
PUBLIC_URL ?=

.PHONY: all install data phase1 phase2-local phase2-docker phase2-public serve

# Sequential recursive calls keep the gates ordered, even with make -j.
all:
	$(MAKE) install
	$(MAKE) data
	$(MAKE) phase1
	$(MAKE) phase2-local
	$(MAKE) phase2-public
	@echo "Phases 1 and 2 verified. Docker deferred by user; phases 3–6 remain unimplemented."

install:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) prepare_data.py

phase1:
	$(PY) -u phase1.py --workers $(WORKERS)
	$(PY) write_results.py

phase2-local:
	$(PY) -c 'import json; assert json.load(open("reports/phase1.json"))["gate"]["passed"], "Phase 1 gate failed"'
	$(PY) -u verify_local.py
	$(PY) write_results.py

phase2-docker:
	$(PY) -u verify_docker.py
	$(PY) write_results.py

phase2-public:
	@test -n "$(PUBLIC_URL)" || (echo "Phase 2 blocked: set PUBLIC_URL to the deployed HTTPS endpoint"; exit 2)
	$(PY) -u verify_public.py --url "$(PUBLIC_URL)"
	$(PY) write_results.py

serve:
	$(PY) serve.py
