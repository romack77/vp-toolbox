
ifeq ($(OS),Windows_NT)
SHELL=cmd
PYTHON=python
VENV_ACTIVATE=.venv\Scripts\activate.bat
else
PYTHON=python3
VENV_ACTIVATE=. .venv/bin/activate
endif

install: .venv

.venv: requirements.txt
	test -d .venv || $(PYTHON) -m virtualenv --always-copy .venv
	($(VENV_ACTIVATE) && pip3 install -Ur requirements.txt && pip3 install -e .)

.ipython-venv: .venv
	($(VENV_ACTIVATE) && pip install -Ur requirements-extras.txt)

ipython: .ipython-venv
	($(VENV_ACTIVATE) && jupyter notebook --port=8889 --no-browser)

test:
	($(VENV_ACTIVATE) && cd vp && $(PYTHON) -m unittest)

# Removes compiled python files.
pyc:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

clean: pyc
	rm -rf .venv

release:
	$(PYTHON) setup.py sdist bdist_wheel
