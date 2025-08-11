## Data Object Creation Performance with Parallel I/O library

This artifact repository accompanies our study on scalable parallel metadata management in parallel I/O libraries. It evaluates three approaches for large-volume data object creation: application-level baseline (with PnetCDF and HDF5 library), library-level baseline and new file format approaches. The latter two approaches require modifications to the I/O library. TThe experiments measure the performance and scalability of parallel data object creation using two Exa.TrkX-derived datasets: **dataset_568k** (568,480 arrays) and **dataset_5m** (5,684,800 arrays, 10× larger) for testing under higher metadata volume. To mimic graph generation workload in our tests, the test program first reads the metadata into memory from a prepared binary input file, and then performs the write operation on a shared output file using parallel I/O library, starting from data object creation.

### Build instructions
#### Install I/O libraries
This project requires the standard HDF5 and PnetCDF libraries, as well as custom variants of PnetCDF developed for this study. The following sections provide detailed instructions for building these I/O libraries.

1. HDF5 1.14.4-2
    Make sure you have a pre-built MPI complier (MPICH or OpenMPI) for parallel HDF5.
    ```shell
     # download source codes
     wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.4.2/hdf5-1.14.4-2.tar.gz
     tar -xf hdf5-1.14.4-2.tar.gz
     cd hdf5-1.14.4-2

     # prefix of install dir. One should modify its value.
     export HDF5_DIR=$HOME/hdf5/HDF5-install

     # configure
     CC=/path/to/your/mpi/bin/mpicc ./configure --prefix=${HDF5_DIR} --enable-parallel --enable-build-mode=production
    
     # compile
     make -j8

     # install
     make install
    ```

2. PnetCDF 1.14.0
   
   Please refer to the instructions in [main README](../README.md) for required softwares (MPI and GNU Autotools) for PnetCDF.
    ```shell
     # make sure you are on pdsw_2025 branch
     git checkout pdsw_2025
     # navigate to the root directory of the repository

     # prefix of install dir. One should modify its value.
     export PNETCDF_DIR=$HOME/pnetcdf/pnetcdf-install
     # configure
     autoreconf -i
     ./configure --prefix=${PNETCDF_DIR} --disable-fortran --disable-cxx CC=cc --enable-shared=no CFLAGS="-Wall -O2"
    
     # compile
     make -j8

     # install
     make install
    ```

3. Library-level Baseline Approach Implementation on PnetCDF 1.14.0
    ```shell
     # make sure you are on pdsw_2025_lib branch
     git checkout pdsw_2025_lib

     # navigate to the root directory of the repository

     # prefix of install dir. One should modify this value.
     export PNETCDF_DIR_LIB=$HOME/pnetcdf/pnetcdf-lib-install
     # configure
     autoreconf -i
     ./configure --prefix=${PNETCDF_DIR_LIB} --disable-fortran --disable-cxx CC=cc --enable-shared=no CFLAGS="-Wall -O2 -Wno-error"
    
     # compile
     make -j8

     # install
     make install
    ```

4. New Format Approach Implementation on PnetCDF 1.14.0
    ```shell
     # make sure you are on pdsw_new_format branch
     git checkout pdsw_2025_new_format

     # prefix of install dir. One should modify its value.
     export PNETCDF_DIR_FORMAT=$HOME/pnetcdf/pnetcdf-format-install

     # configure
     autoreconf -i
     ./configure --prefix=${PNETCDF_DIR_FORMAT} --disable-fortran --disable-cxx CC=cc --enable-shared=no CFLAGS="-Wall -O2 -Wno-error"
    
     # compile
     make -j8

     # install
     make install
    ```

#### Build data object creation test program
    ```shell
     # make sure you are on pdsw_2025 (or pdsw_2025_lib, pdsw_2025_new_format) branch
     git checkout pdsw_2025
     # specify installation paths of I/O libraries. One should modify the paths
     export PNETCDF_DIR=$HOME/pnetcdf/pnetcdf-install
     export PNETCDF_DIR_LIB=$HOME/pnetcdf/pnetcdf-lib-install
     export PNETCDF_DIR_FORMAT=$HOME/pnetcdf/pnetcdf-format-install
     export HDF5_DIR=$HOME/hdf5/hdf5-install
     # navigate to the root directory of the repository and then cd to pdsw2025 folder
     cd pdsw2025
     # compile and build all test programs
     make all
    ```

### Running the experiments

#### Test and profile data object creation

1. Prepare input metadata files
   First, decompress the source metadata file `dataset_568k_metadata.tar.gz`.  
   To generate a larger dataset, duplicate the base file using the utility program `create_ncopy_binary`.  
   You can simply run `./data.sh` to execute this process and prepare dataset_568k and dataset_5m for the test programs.
    ```shell
    ./create_ncopy_binary <source_file> <duplicated_file> <optional: num_copies>
    ```

2. All executables for data object creation tests (app_baseline_test_all, h5_baseline_test_all, lib_baseline_test_all, new_format_test_all) follow the syntax below. Example scripts are provided in `run_[app/h5/lib/new].sh`. For execution in HPC enviroment, we provide an example SLURB job submission script for application-level data object creation test in `job_app.sh`.
   ```
   mpiexec -n <num_proc> <program_name> <source_metadata_file> <output_file>
   ```
   For PnetCDF-based executables, it is advised to configure hash table size using env variable `nc_hash_size_dim/var` for optimized consistenecy check.
   
3. We measure runtime performance by collecting the end-to-end time, starting after metadata is loaded into memory and ending once all data objects are created and metadata is written to file. The test program profiles the runtime into metadata exchange time, metadata consistency check time, and other costs such as file creation and closing. Timings are collected both at test program level and inside I/O library. An example of the output timing message of new file format approach is shown below:
   ```
    [PnetCDF] End-define Phase Timings (seconds):
    - Metadata Exchange             : 0.000087
    - Consistency check (inter-metadata block and shared metadata block): 0.000271
    - Metadata Write I/O            : 0.042027
    [Application] Data Object Creation Timings (seconds):
    End-to-End                    : 0.363126
    - Metadata Consistency Check (intra-metadata block): 0.213572
    - End-define                    : 0.136024
    - Close                         : 0.012277
   ```

#### Memory footprint tracking for data object creation
Heap memroy tracking at multiple checkpoints during data object creation can be enabled for PnetCDF-based tests (app_baseline_test_all/lib_baseline_test_all/new_format_test_all) by adjusting the following configuration in previous builds. **Note:** Enabling memory tracking will noticeably slow down runtime performance. It can be performed on a small-scale run and does not require an HPC environment.
* Add `--enable-profiling` flag in `./configure` command for PnetCDF installation to track memory usage in I/O library
* Add `MEM_TRACK=1` flag in `make` command for test program build to track memory usage in application.

#### Test and profile metadata read from file

1. Perform data object creation and write out to file using previous write tests: app_baseline_test_all (classic netCDF format) and new_format_test_all (new header format)
2. The executables for metadata read tests (app_baseline_read_test_all, new_format_read_test_all) follow the syntax below. The file generated during data object creation serves as the input argument. For execution in HPC enviroment, we provide an example SLURB job submission script for new header format test in `job_new_read.sh`.
   ```
   mpiexec -n <num_proc> <program_name> <created_file>
   ```
   An example output timing message that collects end-to-end read time is shown below:
   ```
   [Application] Metadata Read Time (seconds):
    End-to-End: 4.166268
   ```

