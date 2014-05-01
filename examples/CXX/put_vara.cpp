/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id$ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example shows how to use NcmpiVar::putVar_all() to write a 2D
 * 4-byte integer array in parallel. It first defines a netCDF variable of
 * size global_nx * global_ny where
 *    global_ny == NY and
 *    global_nx == (NX * number of MPI processes).
 * The data partitioning pattern is a column-wise partitioning across all
 * proceses. Each process writes a subarray of size ny * nx.
 *
 *    To compile:
 *        mpicxx -O2 put_vara.cpp -o put_vara -lpnetcdf
 *
 * Example commands for MPI run and outputs from running ncmpidump on the
 * NC file produced by this example program:
 *
 *    % mpiexec -n 4 ./put_vara /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *            y = 10 ;
 *            x = 16 ;
 *    variables:
 *            int var(y, x) ;
 *                var:str_att_name = "example attribute of type text." ;
 *                var:float_att_name = 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f ;
 *    // global attributes:
 *                :history = "Wed Apr 30 11:18:58 2014\n",
 *       "" ;
 *    data:
 *
 *     var =
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 ;
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

#include <string.h>
#include <time.h>   /* time() localtime(), asctime() */

#include <pnetcdf>

using namespace PnetCDF;
using namespace PnetCDF::exceptions;

#define NY 10
#define NX 4

int main(int argc, char** argv) {
    char filename[128], str_att[128];
    int i, j, rank, nprocs, buf[NY][NX];
    float float_att[100];
    MPI_Offset  global_ny, global_nx;

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
        NcmpiFile ncFile(MPI_COMM_WORLD, filename, NcmpiFile::replace,
                         NcmpiFile::data64bits);

        /* the global array is NY * (NX * nprocs) */
        global_ny = NY;
        global_nx = NX * nprocs;

        for (i=0; i<NY; i++)
            for (j=0; j<NX; j++)
                 buf[i][j] = rank;

        /* add a global attribute: a time stamp at rank 0 */
        time_t ltime = time(NULL); /* get the current calendar time */
        asctime_r(localtime(&ltime), str_att);

        /* make sure the time string are consistent among all processes */
        MPI_Bcast(str_att, strlen(str_att), MPI_CHAR, 0, MPI_COMM_WORLD);

        ncFile.putAtt(string("history"), string(str_att));

        /* define dimensions Y and X */
        vector<NcmpiDim> dimid(2);

        dimid[0] = ncFile.addDim("Y", global_ny);
        dimid[1] = ncFile.addDim("X", global_nx);

        /* define a 2D variable of integer type */
        NcmpiVar var = ncFile.addVar("var", ncmpiInt, dimid);

        /* add attributes to the variable */
        var.putAtt(string("str_att_name"),
                   string("example attribute of type text."));

        for (i=0; i<8; i++) float_att[i] = i;
        var.putAtt(string("float_att_name"), ncmpiFloat, 8, float_att);

        /* now we are in data mode */
        vector<MPI_Offset> start(2), count(2);
        start[0] = 0;
        start[1] = NX * rank;
        count[0] = NY;
        count[1] = NX;

        var.putVar_all(start, count, &buf[0][0]);

        /* file is close implicitly */
    }
    catch(NcmpiException& e) {
       cout << e.what() << " error code=" << e.errorCode() << " Error!\n";
       return 1;
    }

    MPI_Finalize();
    return 0;
}

