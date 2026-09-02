-include .env
export

CHANGELOG_DIR=$(PWD)/db/changelog
GENERATED_DIR=$(PWD)/db/changelog

DB_JDBC_URL=jdbc:postgresql://host.docker.internal:5432/vectordb
COMPOSE_PROFILES=crawl4ai

test:
	pytest

sync:
	uv sync --all-groups
install:
	uv pip install -r pyproject.toml

initialize:
	docker compose up
start:
	docker compose start
stop:
	docker compose stop
down:
	docker compose down
destroy:
	docker compose down -v

generate-sql:
	docker run --rm \
	    --add-host host.docker.internal:172.17.0.1 \
	    -e INSTALL_POSTGRESQL=true \
		-v $(CHANGELOG_DIR):/liquibase/changelog \
		liquibase/liquibase \
		--driver=org.postgresql.Driver \
		--changelog-file="./db.changelog-1.0.yaml" \
		--classpath="/liquibase/changelog" \
		--url=$(DB_JDBC_URL) \
		--username=$(DB_USER) \
		--password=$(DB_PASS) \
		updateSQL > $(GENERATED_DIR)/changelog.sql

validate-sql:
	docker run --rm \
	    --add-host host.docker.internal:172.17.0.1 \
	    -e INSTALL_POSTGRESQL=true \
		-v $(CHANGELOG_DIR):/liquibase/changelog \
		liquibase/liquibase \
		--driver=org.postgresql.Driver \
		--changelog-file="./db.changelog-1.0.yaml" \
		--classpath="/liquibase/changelog" \
		--url=$(DB_JDBC_URL) \
		--username=$(DB_USER) \
		--password=$(DB_PASS) \
		validate

status-sql:
	docker run --rm \
	    --add-host host.docker.internal:172.17.0.1 \
	    -e INSTALL_POSTGRESQL=true \
		-v $(CHANGELOG_DIR):/liquibase/changelog \
		liquibase/liquibase \
		--driver=org.postgresql.Driver \
		--changelog-file="./db.changelog-1.0.yaml" \
		--classpath="/liquibase/changelog" \
		--url=$(DB_JDBC_URL) \
		--username=$(DB_USER) \
		--password=$(DB_PASS) \
		status

migrate:
	docker run --rm \
	    --add-host host.docker.internal:172.17.0.1 \
	    -e INSTALL_POSTGRESQL=true \
		-v $(CHANGELOG_DIR):/liquibase/changelog \
		liquibase/liquibase \
		--driver=org.postgresql.Driver \
		--changelog-file="./db.changelog-1.0.yaml" \
		--classpath="/liquibase/changelog" \
		--url=$(DB_JDBC_URL) \
		--username=$(DB_USER) \
		--password=$(DB_PASS) \
		update

run:
	uv run agents.py

run_rag:
	uv run rag.py

run_raptor:
	uv run  rag_raptor.py
