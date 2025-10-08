export PYTHONPATH := src

.PHONY: setup test train optimize serve assets

setup:
	python -m pip install -U pip && pip install -r requirements.txt

test:
	pytest -q

train:
	python -m main demo-train

optimize:
	python -m main optimize

serve:
	python -m main serve

assets:
\tPYTHONPATH=src python scripts/make_readme_assets.py

assets:
	PYTHONPATH=src python scripts/make_readme_assets.py

.PHONY: docker-build docker-run docker-compose docker-down

docker-build:
	docker build -t health-sepsis:latest .

docker-run:
	docker run --rm -p 8000:8000 -e PYTHONPATH=/app/src health-sepsis:latest

docker-compose:
	docker compose up --build

docker-down:
	docker compose down

.PHONY: docker-run-mount docker-build-with-artifacts

docker-run-mount:
	docker run --rm -p 8000:8000 -e PYTHONPATH=/app/src \
		-v "$(PWD)/artifacts:/app/artifacts:ro" health-sepsis:latest

docker-build-with-artifacts:
	# ensure artifacts exist locally
	PYTHONPATH=src python -m main demo-train
	# un-comment COPY line if still commented
	perl -pi -e 's/^# COPY artifacts/COPY artifacts/' Dockerfile
	docker build -t health-sepsis:with-artifacts .
