# Define the name of the virtual environment directory
VENV := venv

# Define the python and pip paths inside the venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: all
all: install

# Create the virtual environment=
$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

# Install dependencies
.PHONY: install
install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

# Run the py script
# usage: make run script=main.py
.PHONY: run
run: install
	$(PYTHON) $(script)

# Clean up: Remove the virtual environment
.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete