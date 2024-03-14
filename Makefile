# For Windows:
# PYTHON = py
# For Windows through Microsoft Store:
# PYTHON = python
# For Mac/Linux (less ambiguous, as `python` may be Python 2):
PYTHON = python3

# For Windows:
# PIP = pip
# For Mac/Linux:
PIP = pip3

# For Windows:
# POETRY = C:\Users\ariel\AppData\Roaming\Python\Scripts\poetry
# For Mac/Linux:
POETRY = poetry

.PHONY: createvenv
createvenv:
	$(PYTHON) -m venv .venv
	$(POETRY) run $(PIP) install --upgrade pip
	$(POETRY) run $(POETRY) install

.PHONY: activatevenv
activatevenv:
	$(POETRY) shell

.PHONY: black
black:
	$(POETRY) run black src --check

.PHONY: black!
black!:
	$(POETRY) run black src

.PHONY: isort
isort:
	$(POETRY) run isort src --check

.PHONY: isort!
isort!:
	$(POETRY) run isort src

.PHONY: format!
format!: black! isort!
