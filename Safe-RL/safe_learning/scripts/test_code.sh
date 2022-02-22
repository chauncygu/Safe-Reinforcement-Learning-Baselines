#!/usr/bin/env bash

module="safe_learning"

get_script_dir () {
     SOURCE="${BASH_SOURCE[0]}"
     # While $SOURCE is a symlink, resolve it
     while [ -h "$SOURCE" ]; do
          DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
          SOURCE="$( readlink "$SOURCE" )"
          # If $SOURCE was a relative symlink (so no "/" as prefix, need to resolve it relative to the symlink base directory
          [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
     done
     DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
     echo "$DIR"
}

# Change to script root
cd $(get_script_dir)/..
GREEN='\033[0;32m'
NC='\033[0m'

# Run style tests
echo -e "${GREEN}Running style tests.${NC}"
flake8 $module --exclude test*.py,__init__.py --ignore=E402,E731,W503 --show-source || { exit 1; }

# Ignore import errors for __init__ and tests
flake8 $module --filename=__init__.py,test*.py --ignore=F,E402,W503 --show-source || { exit 1; }

echo -e "${GREEN}Testing docstring conventions.${NC}"
# Test docstring conventions
pydocstyle $module --convention=numpy || { exit 1; }

# Run unit tests
echo -e "${GREEN}Running unit tests.${NC}"
pytest --doctest-modules --cov --cov-fail-under=80 $module || { exit 1; }

