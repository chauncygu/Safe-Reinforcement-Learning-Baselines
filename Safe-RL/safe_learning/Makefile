.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

doc: ## Build documentation (docs/_build/html/index.html)
	cd docs && $(MAKE) html

coverage: ## Construct coverage (htmlcov/index.html)
	coverage html

test-local: ## Test the local installation of the code
	./scripts/test_code.sh

test: docker ## Test the docker images
	docker run safe_learning_py2 make test-local
	docker run safe_learning_py3 make test-local

dev: ## Mount current code as volume and run jupyterlab for development
	docker build -f Dockerfile.dev -t safe_learning_dev .
	docker run -p 8888:8888 -v $(shell pwd):/code safe_learning_dev

docker: ## Build the docker images
	docker build -f Dockerfile.python2 -t safe_learning_py2 .
	docker build -f Dockerfile.python3 -t safe_learning_py3 .

