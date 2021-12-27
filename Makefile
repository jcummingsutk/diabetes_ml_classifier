install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C *.py
	pylint --disable=R,C src/utils/*.py

test:
	python -m pytest -vv --cov=hello *.py

all: install lint test
