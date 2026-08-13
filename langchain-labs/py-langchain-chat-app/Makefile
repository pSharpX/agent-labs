-include .env
export

CHANGELOG_DIR=$(PWD)/db/changelog
GENERATED_DIR=$(PWD)/db/changelog

DB_JDBC_URL=jdbc:postgresql://host.docker.internal:5432/vectordb

test:
	pytest

install:
	pip install -r requirements.txt

initialize:
	docker compose up

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
	python main.py
run2:
	python imperative.py
run3:
	python declarative.py
run4:
	python chat.py
run5:
	python generic_chat.py

run6:
	python tools.py

run7:
	python rag.py
run8:
	python vector_store.py