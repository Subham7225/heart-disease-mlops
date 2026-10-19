PYTHON=python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) data/download_data.py

train:
	$(PYTHON) -m src.train --config configs/default.yaml

test:
	pytest -q

lint:
	ruff check .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000
