## This can be added to a pre-commit hook to ensure that the wrappers build
## successfully before each commit
all:
	swig -python -c++ -o topology_wrap.cpp topology.i
	mv topology.py topopy/topology.py

clean:
	rm topology_wrap.cpp topopy/topology.py
