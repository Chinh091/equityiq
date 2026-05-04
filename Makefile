.PHONY: help install up down logs ps lint typecheck test fmt smoke models

help:
	@echo "EquityIQ — common targets"
	@echo "  install     - sync uv workspace deps"
	@echo "  up          - start docker-compose stack"
	@echo "  down        - stop stack (keep volumes)"
	@echo "  logs        - tail compose logs"
	@echo "  ps          - compose status"
	@echo "  models      - pull Ollama models into container"
	@echo "  fmt         - ruff format + fix"
	@echo "  lint        - ruff check"
	@echo "  typecheck   - mypy"
	@echo "  test        - pytest (excludes integration)"
	@echo "  test-int    - pytest (integration only; needs services up)"
	@echo "  smoke       - end-to-end smoke against /thesis/stream"
	@echo "  api         - run FastAPI dev server"

install:
	uv sync --all-packages

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

models:
	docker compose exec ollama ollama pull $${OLLAMA_PRIMARY_MODEL:-llama3.3:70b-instruct-q4_K_M}
	docker compose exec ollama ollama pull $${OLLAMA_FALLBACK_MODEL:-qwen2.5:32b-instruct}
	docker compose exec ollama ollama pull $${OLLAMA_EMBED_MODEL:-nomic-embed-text:v1.5}

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy packages/*/src apps/*/src

test:
	uv run pytest -m "not integration"

test-int:
	uv run pytest -m integration

api:
	uv run uvicorn equityiq_api.main:app --reload --host 0.0.0.0 --port 8000

smoke:
	curl -N -X POST http://localhost:8000/thesis/stream \
	  -H 'content-type: application/json' \
	  -d '{"ticker":"NVDA","question":"summarize recent risk factors"}'
