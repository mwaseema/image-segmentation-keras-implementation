all: install clean

install:
	python setup.py install
clean:
	python setup.py clean
