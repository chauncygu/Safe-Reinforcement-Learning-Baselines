module="SafeRLBench"

GREEN=\033[0;32m
NC=\033[0m

# Flake 8 ignore errors
flakeignore='E402,W503'

# Pydocstyle ignore errors
pydocignore='D105'

style:
	@echo "${GREEN}Running style tests:${NC}"
	@flake8 ${module} --exclude test*.py,__init__.py --show-source
	@flake8 ${module} --filename=__init__.py,test*.py --ignore=F --show-source

docstyle:
	@echo "${GREEN}Testing docstring conventions:${NC}"
	@pydocstyle ${module} --match='(?!__init__).*\.py' 2>&1 | grep -v "WARNING: __all__"

unittests:
	@echo "${GREEN}Running unit tests in current environment.${NC}"
	@nosetests -v --with-doctest --with-coverage --cover-erase --cover-package=${module} ${module}  2>&1 | grep -v "^Level 1"

coverage: unittests
	@echo "${GREEN}Create coverage report:${NC}"
	@coverage html

test: style docstyle unittests

# targets to setup docker images for testing
setup_docker2:
	docker build -f misc/Dockerfile.python2 -t srlb-py27-image .

setup_docker3:
	docker build -f misc/Dockerfile.python3 -t srlb-py35-image .

setup_docker: setup_docker2 setup_docker3

docker2:
	@echo "${GREEN}Running unit tests for 2.7 in docker container:${NC}"
	@docker run -e "TF_CPP_MIN_LOG_LEVEL=2" -v $(shell pwd):/code/ srlb-py27-image nosetests --with-doctest --verbosity=2 SafeRLBench  2>&1 | grep -v "^Level "

docker3:
	@echo "${GREEN}Running unit tests for 3.5 in docker container:${NC}"
	@docker run -e "TF_CPP_MIN_LOG_LEVEL=2" -v $(shell pwd):/code/ srlb-py35-image nosetests --with-doctest --verbosity=2 SafeRLBench  2>&1 | grep -v "^Level "

docker: docker2 docker3

history:
	git log --graph --decorate --oneline

clean:
	find . -type f -name '*.pyc' -exec rm -f {} ';'
	rm -r htmlcov
