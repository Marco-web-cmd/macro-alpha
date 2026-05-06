.PHONY: dev worker docker-build up up-llm down logs test

# ── Développement local ──────────────────────────────────────
dev:
	source ~/alpha_trading/venv/bin/activate && \
	uvicorn app:app --host 0.0.0.0 --port 5001 --reload

worker:
	source ~/alpha_trading/venv/bin/activate && \
	celery -A tasks worker --loglevel=info --concurrency=1

# ── Docker ───────────────────────────────────────────────────
docker-build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Dashboard : http://localhost:5001"
	@echo "Health    : http://localhost:5001/api/health"

up-llm:
	docker-compose --profile llm up -d
	docker-compose exec ollama ollama pull llama3.1:8b

down:
	docker-compose down

logs:
	docker-compose logs -f api worker

# ── Tests ─────────────────────────────────────────────────────
test:
	source ~/alpha_trading/venv/bin/activate && python3 -c "\
import ast, os, sys; \
errors = []; \
[errors.append(f) or print(f'ERREUR: {f} — {e}') \
 for root, dirs, files in os.walk('.') \
 for _ in [dirs.__setitem__(slice(None), [d for d in dirs if d not in ['venv', '.git', '__pycache__']])] \
 for f in files if f.endswith('.py') \
 for e in [None] if (lambda: (_ := ast.parse(open(os.path.join(root, f)).read()), print(f'OK: {os.path.join(root, f)}'))() or False] \
]; sys.exit(1 if errors else 0)"
