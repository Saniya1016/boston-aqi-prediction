# Define the name of the virtual environment directory
VENV := venv

# OS Detection & Configuration
ifeq ($(OS),Windows_NT)
    # Windows Settings
    VENV_BIN := $(VENV)/Scripts
    PYTHON_CMD := python
    CLEAN_VENV := rmdir /s /q $(VENV)
    CLEAN_CACHE := powershell -c "Get-ChildItem . -Recurse -Directory -Filter '__pycache__' | Remove-Item -Force -Recurse" ; \
                   powershell -c "Get-ChildItem . -Recurse -Include *.pyc -File | Remove-Item -Force"
else
    # Linux / macOS Settings
    VENV_BIN := $(VENV)/bin
    PYTHON_CMD := python3
    CLEAN_VENV := rm -rf $(VENV)
    CLEAN_CACHE := find . -type d -name "__pycache__" -exec rm -rf {} + ; \
                   find . -type f -name "*.pyc" -delete
endif

# Paths & Variables
PYTHON  := $(VENV_BIN)/python
PIP     := $(VENV_BIN)/pip
JUPYTER := $(VENV_BIN)/jupyter

# Notebook Paths
AQI_NOTEBOOK    := Models/aqi_model/aqi_predict_cts.ipynb
POLLEN_NOTEBOOK := Models/pollen_models/pollen_models.ipynb
STREAMLIT_APP   := visualizations/Plum/script/streamlit.py

# Targets
.PHONY: all
all: install

# Create the virtual environment
# We use the variable $(VENV_BIN) so it works on both OS types
$(VENV_BIN)/activate:
	$(PYTHON_CMD) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip

# Install dependencies
.PHONY: install
install: $(VENV_BIN)/activate
	$(PYTHON) -m pip install -r requirements.txt

# Execution Targets

## Download dataset from Hugging Face
DATA_DIR := Data
AQI_CSV := $(DATA_DIR)/boston_pollutants.csv
WEATHER_CSV := $(DATA_DIR)/boston_weather.csv
POLLEN_CSV := $(DATA_DIR)/boston_pollen.csv

.PHONY: download-data
download-data: $(AQI_CSV) $(WEATHER_CSV) $(POLLEN_CSV)

$(AQI_CSV):
	mkdir -p $(DATA_DIR)
	curl -L "https://huggingface.co/datasets/saneya-singh/boston-aqi-pollen/resolve/main/boston_pollutants_with_aqi_include_2024.csv" -o $@

$(WEATHER_CSV):
	mkdir -p $(DATA_DIR)
	curl -L "https://huggingface.co/datasets/saneya-singh/boston-aqi-pollen/resolve/main/boston-weather-data(open_meteo).csv" -o $@

$(POLLEN_CSV):
	mkdir -p $(DATA_DIR)
	curl -L "https://huggingface.co/datasets/saneya-singh/boston-aqi-pollen/resolve/main/EPHT_Pollen_Data.csv" -o $@


## Run the AQI Prediction Notebook
.PHONY: run-aqi-nb
run-aqi-nb: install download-data
	@echo "Running AQI Prediction Notebook..."
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(AQI_NOTEBOOK)

## Run the Pollen Models Notebook
.PHONY: run-pollen-nb
run-pollen-nb: install download-data
	@echo "Running Pollen Models Notebook..."
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(POLLEN_NOTEBOOK)

## Run the Streamlit Application
.PHONY: run-streamlit
run-streamlit: install download-data
	@echo "Starting Streamlit App..."
	$(PYTHON) -m streamlit run $(STREAMLIT_APP)

## Run All Targets
.PHONY: run
run: run-aqi-nb run-pollen-nb run-streamlit
	@echo "✅ All required processes have been started/executed."

# Cleanup (Optional)

# .PHONY: clean
# clean:
# 	@echo "Cleaning up environment and cache..."
# 	-$(CLEAN_VENV)
# 	-$(CLEAN_CACHE)