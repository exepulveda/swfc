all: clusteringlib.so

clusteringlib.so: utils.o distances.o clustering.o sfc_ga.o  clusteringlib.pyf
	f2py -c clusteringlib.pyf utils.o distances.o clustering.o sfc_ga.o --f90flags="-Ofast -fPIC" --fcompiler="gnu95"

clustering.o: utils.f90 distances.f90 clustering.f90 
	gfortran -c clustering.f90 -fPIC

distances.o: distances.f90
	gfortran -c distances.f90 -fPIC

utils.o: utils.f90
	gfortran -c utils.f90 -fPIC

sfc_ga.o: utils.f90 distances.f90 clustering.f90 sfc_ga.f90
	gfortran -c sfc_ga.f90 -fPIC

clusteringlib.pyf: utils.f90 distances.f90 clustering.f90  sfc_ga.f90 
	f2py utils.f90 distances.f90 clustering.f90 sfc_ga.f90  -m clusteringlib -h clusteringlib.pyf --overwrite-signature

clean:
	rm -f *.pyf *.so *.o *.mod 
