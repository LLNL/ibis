SHELL := /bin/bash

USER_WORKSPACE := $(if $(USER_WORKSPACE),$(USER_WORKSPACE),/usr/workspace/$(USER))
WORKSPACE = $(USER_WORKSPACE)/gitlab/weave/ibis
IBIS_ENV := $(if $(IBIS_ENV),$(IBIS_ENV),ibis_env)


PYTHON_CMD = /usr/tce/packages/python/python-3.8.2/bin/python3

PIP_OPTIONS = --trusted-host wci-repo.llnl.gov --index-url https://wci-repo.llnl.gov/repository/pypi-group/simple --use-pep517

DOCS_PKGS = sphinx nbsphinx nbconvert sphinx-rtd-theme

# TEMPORARY till we get trata to Pypi or wci-repo
CZ_GITLAB = ssh://git@czgitlab.llnl.gov:7999
RZ_GITLAB = ssh://git@rzgitlab.llnl.gov:7999
SCF_GITLAB = ssh://git@scfgitlab.llnl.gov:7999

# SOURCE_ZONE is set in CI variable
# Set SOURCE_ZONE to 'CZ', 'RZ' or 'SCF' on command line for manual testing
ifeq ($(SOURCE_ZONE),SCF)
    GITLAB_URL = $(SCF_GITLAB)
else ifeq ($(SOURCE_ZONE),RZ)
    GITLAB_URL = $(RZ_GITLAB)
else
    GITLAB_URL = $(CZ_GITLAB)
endif
TRATA_REPO = $(GITLAB_URL)/weave/trata

BUILDS_DIR := $(if $(CI_BUILDS_DIR),$(CI_BUILDS_DIR)/gitlab/weave/weave_ci,$(shell pwd))

define create_env
	# call from the directory where env will be created
	# arg1: name of env
	$(PYTHON_CMD) -m venv $1
	source $1/bin/activate && \
	which pip && \
	pip install $(PIP_OPTIONS) --upgrade pip && \
	pip install $(PIP_OPTIONS) --upgrade setuptools && \
	pip install $(PIP_OPTIONS) --force pytest
endef

define install_trata
	echo "...install Trata..."
	cd $(WORKSPACE) && \
	rm -rf trata && git clone $(TRATA_REPO) && \
	cd trata  && \
	source $1/bin/activate && \
	pip install $(PIP_OPTIONS) . && \
	pip list
endef

define install_ibis
	cd $(BUILDS_DIR)
	source $1/bin/activate && \
	pip install $(PIP_OPTIONS) .  && \
	pip list && which pip
endef


define run_ibis_tests
	# call from the top repository directory
	# arg1: full path to venv
	source $1/bin/activate && \
	which pytest && \
	if [ $(TESTS) ]; then \
		pytest --capture=tee-sys -v $(TESTS); \
	else \
		pytest --capture=tee-sys -v tests/; \
	fi
endef


.PHONY: create_env
create_env:
	@echo "Create venv for running ibis...$(WORKSPACE)";
	@[ -d $(WORKSPACE) ] || mkdir -p $(WORKSPACE);
	cd $(WORKSPACE)
	if [ -d $(IBIS_ENV) ]; then \
		rm -rf $(IBIS_ENV); \
	fi
	$(call create_env,$(WORKSPACE)/$(IBIS_ENV))
	if [ -d trata ]; then \
	  rm -rf trata; \
	fi;
	$(call install_trata,$(WORKSPACE)/$(IBIS_ENV))
	$(call install_ibis,$(WORKSPACE)/$(IBIS_ENV))

.PHONY: run_tests
run_tests:
	@echo "Run tests...";
	$(call run_ibis_tests,$(WORKSPACE)/$(IBIS_ENV))


.PHONY: build_docs
build_docs:
	@echo "Build docs...";
	source $(WORKSPACE)/$(IBIS_ENV)/bin/activate && \
	pip install $(PIP_OPTIONS) $(DOCS_PKGS) && \
	cd docs && make html


