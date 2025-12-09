# Define the name of the virtual environment directory
VENV := venv

# Define the paths inside the venv for Windows/PowerShell (using forward slashes for Make)
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip
JUPYTER := $(VENV)/Scripts/jupyter

#  File Paths 

# Notebook paths (Note: Forward slashes are used for cross-platform compatibility in Make)
AQI_NOTEBOOK := Models/aqi_model/aqi_predict_cts.ipynb
POLLEN_NOTEBOOK := Models/pollen_models/pollen_models.ipynb

# Streamlit application path
STREAMLIT_APP := visualizations/Plum/script/streamlit.py

# Default target
.PHONY: all
all: install

# Create the virtual environment (Using 'python' instead of 'python3' for Windows reliability)
$(VENV)/Scripts/activate:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip

# Install dependencies
.PHONY: install
install: $(VENV)/Scripts/activate
	$(PYTHON) -m pip install -r requirements.txt

#  Execution Targets 

## Run the AQI Prediction Notebook
# Command: make run-aqi-nb
.PHONY: run-aqi-nb
run-aqi-nb: install
	@echo " Running AQI Prediction Notebook: $(AQI_NOTEBOOK) "
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(AQI_NOTEBOOK)

## Run the Pollen Models Notebook
# Command: make run-pollen-nb
.PHONY: run-pollen-nb
run-pollen-nb: install
	@echo " Running Pollen Models Notebook: $(POLLEN_NOTEBOOK) "
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(POLLEN_NOTEBOOK)

## Run the Streamlit Application
# Command: make run-streamlit
.PHONY: run-streamlit
run-streamlit: install
	@echo " Starting Streamlit App: $(STREAMLIT_APP) "
	$(PYTHON) -m streamlit run $(STREAMLIT_APP)

## Run All Targets
# Command: make run
.PHONY: run
run: run-aqi-nb run-pollen-nb run-streamlit
	@echo "✅ All required processes have been started/executed."

#  Clean up (Commented Out) 

# # Clean up: Remove the virtual environment and cache files
# .PHONY: clean
# clean:
# 	@echo Removing virtual environment and cache...
# 	# Use native Windows rmdir command
# 	rmdir /s /q $(VENV)
# 	# Use PowerShell commands for cache files, as they are reliable on Windows
# 	powershell -c "Get-ChildItem . -Recurse -Directory -Filter '__pycache__' | Remove-Item -Force -Recurse"
# 	powershell -c "Get-ChildItem . -Recurse -Include *.pyc -File | Remove-Item -Force"