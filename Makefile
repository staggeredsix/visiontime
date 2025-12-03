
.PHONY: build up down logs models

build:
	docker compose build

models:
	python scripts/download_models.py

up: models
	docker compose up -d

logs:
	docker compose logs -f

down:
	docker compose down
