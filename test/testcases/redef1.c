#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pnetcdf.h>

#define PNCDF_Error(status, msg) \
    if (status != NC_NOERR) { \
        printf("Error: %s (%s)\n", msg, ncmpi_strerror(status)); \
        pass = 0; \
        goto fn_exit; \
    }  

int main(int argc, char** argv)
{
    char filename[128]="redef1.nc";
    int i, j, k, commsize, rank, ncid, verbose=0, status, pass=1;
    int dim0id, dim1id, dim5id, dim9id, dim2id, dimsid[2], dims2id[2];
    int varid, var3id, var4id, var2id;
    int *data;
    double *dbl_data;
    MPI_Offset len0=10, len1=3, len5=5, len9=9, len2=10;
    MPI_Offset start[2], count[2];
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);

    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (argc == 2) strcpy(filename, argv[1]);

    if (commsize > 1 && rank == 0 && verbose)
        printf("Warning: %s is designed to run on 1 process\n",argv[0]);
  
    status = ncmpi_create(comm, filename, NC_CLOBBER|NC_64BIT_OFFSET,
                          MPI_INFO_NULL, &ncid);
    PNCDF_Error(status, "create")
  
    status = ncmpi_def_dim(ncid, "dim0", len0, &dim0id);
    PNCDF_Error(status, "def_dim0")

    status = ncmpi_def_dim(ncid, "dim1", len1, &dim1id);
    PNCDF_Error(status, "def_dim1")

    status = ncmpi_def_dim(ncid, "dim5", len5, &dim5id);
    PNCDF_Error(status, "def_dim5")

    status = ncmpi_def_dim(ncid, "dim9", len9, &dim9id);
    PNCDF_Error(status, "def_dim9")
  
    dimsid[0] = dim0id;
    dimsid[1] = dim1id;
    status = ncmpi_def_var(ncid, "xyz", NC_INT, 2, dimsid, &varid);
    PNCDF_Error(status, "def_var")
 
    dimsid[0] = dim0id;
    dimsid[1] = dim5id;
    status = ncmpi_def_var(ncid, "connect", NC_INT, 2, dimsid, &var3id);
    PNCDF_Error(status, "def_var3")

    dimsid[0] = dim0id;
    dimsid[1] = dim9id;
    status = ncmpi_def_var(ncid, "connect_exterior", NC_INT, 2, dimsid, &var4id);
    PNCDF_Error(status, "def_var4")

    status = ncmpi_enddef(ncid);
    PNCDF_Error(status, "enddef")

    //put data
    start[0] = 0;
    start[1] = 0;
    count[0] = len0;
    count[1] = len1;

    data = (int*) malloc(len0*len1 * sizeof(int));
    k=0;
    for (i=0; i<len0; i++)
        for (j=0; j<len1; j++)
            data[i*len1+j] = k++;
    if (rank > 0) count[0] = count[1] = 0;
    status = ncmpi_put_vara_int_all(ncid, varid, start, count, &data[0]);
    PNCDF_Error(status, "put1")
    free(data);
    
    count[0] = len0;
    count[1] = len5;
    data = (int*) malloc(len0*len5 * sizeof(int));
    k=0;
    for (i=0; i<len0; i++)
        for (j=0; j<len5; j++)
            data[i*len5+j] = k++;
    if (rank > 0) count[0] = count[1] = 0;
    status = ncmpi_put_vara_int_all(ncid, var3id, start, count, &data[0]);
    PNCDF_Error(status, "put3")
    free(data);

    count[0] = len0;
    count[1] = len9;
    data = (int*) malloc(len0*len9 * sizeof(int));
    k=0;
    for (i=0; i<len0; i++)
        for (j=0; j<len9; j++)
            data[i*len9+j] = k++;
    if (rank > 0) count[0] = count[1] = 0;
    status = ncmpi_put_vara_int_all(ncid, var4id, start, count, &data[0]);
    PNCDF_Error(status, "put4")
    free(data);

    status = ncmpi_close(ncid);
    PNCDF_Error(status, "close")

    status = ncmpi_open(comm, filename, NC_WRITE, MPI_INFO_NULL, &ncid);

    status = ncmpi_redef(ncid);
    PNCDF_Error(status, "redef")

    status = ncmpi_def_dim(ncid, "dim2", len2, &dim2id);
    PNCDF_Error(status, "def_dim")
  
    dims2id[0] = dim0id;
    dims2id[1] = dim2id;
    status = ncmpi_def_var(ncid, "xyz_r", NC_DOUBLE, 2, dims2id, &var2id);
    PNCDF_Error(status, "def_var")

    status = ncmpi_enddef(ncid);
    PNCDF_Error(status, "enddef")

    start[0] = 0;
    start[1] = 0;
    count[0] = len0;
    count[1] = len2;
    k=0;
    dbl_data = (double*) malloc(len0*len2 * sizeof(double));
    for (i=0; i<len0; i++)
        for (j=0; j<len2; j++) {
            dbl_data[i*len2+j] = (k*k);
            k++;
        }
    if (rank > 0) count[0] = count[1] = 0;
    status = ncmpi_put_vara_double_all(ncid, var2id, start, count, &dbl_data[0]);
    PNCDF_Error(status, "put2")
    free(dbl_data);

    status = ncmpi_close(ncid);
    PNCDF_Error(status, "close")

fn_exit:
    if (rank == 0) {
        char cmd_str[80];
        sprintf(cmd_str, "*** TESTING C   %s for entering re-define mode ", argv[0]);
        if (pass) printf("%-66s ------ pass\n", cmd_str);
        else      printf("%-66s ------ failed\n", cmd_str);
    }

    MPI_Finalize();
    return 0;
}