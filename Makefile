
openblas:
	cd OpenBLAS/; make; make PREFIX=.. install; cd ..;
	
ext:
	python setup.py build_ext -i;

clean_openblas:
	rm -rf include;	rm -rf lib; cd OpenBLAS; make clean; cd ..;

clean:
	rm -rf blasy/*.c blasy/*.so; rm -rf build;

