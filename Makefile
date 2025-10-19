.PHONY: all sqush

all: build

build:
	./build.sh

sqush: clean
	./enroot.sh -n cuda -f ${PWD}/Dockerfile

clean:
	rm -rf build/
