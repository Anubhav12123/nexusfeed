.PHONY: install install-dev docker-up docker-down migrate seed demo test lint format benchmark load-test train

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

docker-up:
	# ./data is bind-mounted into the api/worker containers, which run as a
	# non-root user (uid 1000). A fresh clone's ./data dir is owned by your
	# host user, so without this the container can't write the FAISS index,
	# ranking model, or MLflow sqlite db to it (silent-ish "unable to open
	# database file" / index-not-found errors, not an obvious permissions
	# message).
	mkdir -p data && chmod 777 data
	docker compose up -d --build
	@echo "API:        http://localhost:8000/docs"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000 (admin/nexusfeed)"

docker-down:
	docker compose down -v

migrate:
	alembic upgrade head

seed:
	python -m scripts.generate_synthetic_data --users 2000 --items 5000 --interactions 50000
	python -m scripts.seed_faiss_index

demo:
	python -m scripts.demo

benchmark:
	python -m scripts.run_benchmark

train:
	python -m training.trainer --data ./data/interactions.parquet --num-users 2000 --num-items 5000 --num-categories 10

test:
	pytest tests/unit -v --cov=nexusfeed --cov-report=term-missing

test-integration:
	pytest tests/integration -v -m integration

load-test:
	locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 5m --host http://localhost:8000

lint:
	flake8 nexusfeed training tests
	mypy nexusfeed

format:
	isort nexusfeed training tests scripts
	black nexusfeed training tests scripts
