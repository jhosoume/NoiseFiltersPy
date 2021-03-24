PACKAGE := NoiseFiltersPy

all: clean install-complete code-check
.PHONE: all clean code-check pypi install-complete install

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	rm -rf dist
	rm -rf build
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

code-check:
	flake8 $(PACKAGE)
	pylint $(PACKAGE) -j 0 -d 'C0103, R0913, R0902, R0914, C0302, R0904, R0801, E1101, C0330, E1136'
	mypy $(PACKAGE) --ignore-missing-imports

pypi: clean ## Send the package to pypi.
	pip install -U twine wheel
	python3 setup.py sdist bdist_wheel
	twine upload dist/*

install-complete: 
	pip install -U -e .
	pip install -U -r requirements.txt
	pip install -U -r requirements-dev.txt
	pip install -U -r requirements-docs.txt

install:
	pip install .