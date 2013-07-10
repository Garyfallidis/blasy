
openblas:
	cd OpenBLAS/; make; make PREFIX=.. install; cd ..;

openblas_omp:
	cd OpenBLAS/; export OMP_NUM_THREADS=8; make USE_OPENMP=1; make PREFIX=.. install; cd ..;	
	
ext:
	python setup.py build_ext -i;

clean_openblas:
	rm -rf include;	rm -rf lib; cd OpenBLAS; make clean; cd ..;

clean:
	rm -rf blasy/*.c blasy/*.so; rm -rf build;

