export LD_PRELOAD="/scratch/yll6162/romio/romio-install/lib/libromio.so:/scratch/yll6162/pnetcdf/pnetcdf-install/lib/libpnetcdf.so"
mpiexec -n 16 ./build/collective_write_cu collective_write_cu.nc 100