.PHONY: install proto dev docker-up docker-down docker-logs

install:
	uv pip install -e .

proto:
	python scripts/generate_proto.py

dev:
	python -m scraper.main

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f web-scraper


