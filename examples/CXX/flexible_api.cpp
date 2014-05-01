/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This example shows how to use PnetCDF flexible APIs, NcmpiVar::putVar_all()
 * to write two 2D array variables (one is of 4-byte integer byte and the
 * other float type) in parallel. It first defines 2 netCDF variables of sizes
 *    var_zy: NZ*nprocs x NY
 *    var_yx: NY x NX*nprocs
 *
 * The data partitioning patterns on the 2 variables are row-wise and
 * column-wise, respectively. Each process writes a subarray of size
 * NZ x NY and NY x NX to var_zy and var_yx, respectively.
 * Both local buffers have a ghost cell of length 3 surrounded along each
 * dimension.
 *
 * The compile and run commands are given below.
 *
 *    % mpicxx -O2 -o flexible_api flexible_api.cpp -lpnetcdf
 *
 *    % mpiexec -l -n 4 ./flexible_api /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *            Z = 20 ;
 *            Y = 5 ;
 *            X = 20 ;
 *    variables:
 *            int var_zy(Z, Y) ;
 *            float var_yx(Y, X) ;
 *    data:
 *
 *     var_zy =
 *      0, 0, 0, 0, 0,
 *      0, 0, 0, 0, 0,
 *      0, 0, 0, 0, 0,
 *      0, 0, 0, 0, 0,
 *      0, 0, 0, 0, 0,
 *      1, 1, 1, 1, 1,
 *      1, 1, 1, 1, 1,
 *      1, 1, 1, 1, 1,
 *      1, 1, 1, 1, 1,
 *      1, 1, 1, 1, 1,
 *      2, 2, 2, 2, 2,
 *      2, 2, 2, 2, 2,
 *      2, 2, 2, 2, 2,
 *      2, 2, 2, 2, 2,
 *      2, 2, 2, 2, 2,
 *      3, 3, 3, 3, 3,
 *      3, 3, 3, 3, 3,
 *      3, 3, 3, 3, 3,
 *      3, 3, 3, 3, 3,
 *      3, 3, 3, 3, 3 ;
 *
 *     var_yx =
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3 ;
 *    }
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

#include <string.h>
#include <pnetcdf>

using namespace PnetCDF;
using namespace PnetCDF::exceptions;

#define NZ 5
#define NY 5
#define NX 5

int main(int argc, char** argv) {
    char filename[128];
    int i, rank, nprocs, err, ghost_len=3;
    int ncid, cmode, varid0, varid1, dimid[3], *buf_zy;
    float *buf_yx;
    MPI_Datatype  subarray;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (argc == 2) strcpy(filename, argv[1]);
    else           strcpy(filename, "testfile.nc");

    try {
        /* create a new file for writing ------------------------------------*/
        NcmpiFile nc(MPI_COMM_WORLD, filename, NcmpiFile::replace,
                     NcmpiFile::data64bits);

        /* define 3 dimensions */
        vector<NcmpiDim> dimid(3);
        dimid[0] = nc.addDim("Z", NZ*nprocs);
        dimid[1] = nc.addDim("Y", NY);
        dimid[2] = nc.addDim("X", NX*nprocs);

        vector<NcmpiDim> dimid0(2), dimid1(2);
        dimid0[0] =             dimid[0];
        dimid0[1] = dimid1[0] = dimid[1];
        dimid1[1] =             dimid[2];

        /* define a variable of size (NZ * nprocs) * NY */
        NcmpiVar var0 = nc.addVar("var_zy", ncmpiInt,   dimid0);

        /* define a variable of size NY * (NX * nprocs) */
        NcmpiVar var1 = nc.addVar("var_yx", ncmpiFloat, dimid1);

        /* var_zy is partitioned along Z dimension */
        int array_of_sizes[2], array_of_subsizes[2], array_of_starts[2];
        array_of_sizes[0]    = NZ + 2*ghost_len;
        array_of_sizes[1]    = NY + 2*ghost_len;
        array_of_subsizes[0] = NZ;
        array_of_subsizes[1] = NY;
        array_of_starts[0]   = ghost_len;
        array_of_starts[1]   = ghost_len;
        MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                                 array_of_starts, MPI_ORDER_C, MPI_INT,
                                 &subarray);
        MPI_Type_commit(&subarray);

        /* allocate buffer buf_zy and intialize its contents */
        int buffer_len = (NZ+2*ghost_len) * (NY+2*ghost_len);
        buf_zy = (int*) malloc(buffer_len * sizeof(int));
        for (i=0; i<buffer_len; i++) buf_zy[i] = rank;

        vector <MPI_Offset> start(2), count(2);
        start[0] = NZ * rank; start[1] = 0;
        count[0] = NZ;        count[1] = NY;
        /* calling a blocking flexible API */
        var0.putVar_all(start, count, &buf_zy[0], 1, subarray);
        free(buf_zy);

        /* var_yx is partitioned along X dimension */
        array_of_sizes[0]    = NY + 2*ghost_len;
        array_of_sizes[1]    = NX + 2*ghost_len;
        array_of_subsizes[0] = NY;
        array_of_subsizes[1] = NX;
        array_of_starts[0]   = ghost_len;
        array_of_starts[1]   = ghost_len;
        MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                                 array_of_starts, MPI_ORDER_C, MPI_FLOAT,
                                 &subarray);
        MPI_Type_commit(&subarray);

        /* allocate buffer buf_yx and intialize its contents */
        buffer_len = (NY+2*ghost_len) * (NX+2*ghost_len);
        buf_yx = (float*) malloc(buffer_len * sizeof(float));
        for (i=0; i<buffer_len; i++) buf_yx[i] = rank;

        start[0] = 0;  start[1] = NX * rank;
        count[0] = NY; count[1] = NX;

        /* calling a non-blocking flexible API */
        var1.putVar_all(start, count, buf_yx, 1, subarray);
        free(buf_yx);

        /* file is close implicitly */
    }
    catch(NcmpiException& e) {
       cout << e.what() << " error code=" << e.errorCode() << " Error!\n";
       return 1;
    }

    MPI_Finalize();
    return 0;
}

