/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the corresponding APIs defined in src/dispatchers/file.c
 *
 * ncmpi_close() : dispatcher->close()
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h> /* open(), lseek() */
#include <sys/stat.h>  /* open() */
#include <fcntl.h>     /* open() */
#include <unistd.h>    /* truncate(), lseek() */
#include <errno.h>

#include <mpi.h>

#include <pnc_debug.h>
#include <common.h>
#include "ncmpio_NC.h"
#ifdef ENABLE_SUBFILING
#include "ncmpio_subfile.h"
#endif

int free_counter = 0;
int cls_counter = 0;
/*----< ncmpio_free_NC() >----------------------------------------------------*/
void
ncmpio_free_NC(NC *ncp)
{
    int rank;
    MPI_Comm_rank(ncp->comm, &rank);
    double start_time = MPI_Wtime();

    if (ncp == NULL) return;
    
    ncmpio_free_NC_dimarray(&ncp->dims);
    // if (rank == 0)
    //     printf("ncmpio_free_NC free() count after ncmpio_free_NC_dimarray: %d\n", free_counter);
    double dim_free_time = MPI_Wtime() - start_time;
    ncmpio_free_NC_attrarray(&ncp->attrs);
    // if (rank == 0)
    //     printf("ncmpio_free_NC free() count after ncmpio_free_NC_attarray: %d\n", free_counter);
    start_time = MPI_Wtime();
    ncmpio_free_NC_vararray(&ncp->vars);
    double other_start = MPI_Wtime();
    
    // if (rank == 0)
    //     printf("ncmpio_free_NC free() count after ncmpio_free_NC_vararray: %d\n", free_counter);
    double var_free_time = MPI_Wtime() - start_time;

    /* The only case that ncp->mpiinfo is MPI_INFO_NULL is when exiting endef
     * from a redef. All other cases reaching here are from ncmpi_close, in
     * which case ncp->mpiinfo is never MPI_INFO_NULL.
     */
    if (ncp->mpiinfo != MPI_INFO_NULL) MPI_Info_free(&ncp->mpiinfo);

    if (ncp->get_list != NULL) NCI_Free(ncp->get_list);
    if (ncp->put_list != NULL) NCI_Free(ncp->put_list);
    if (ncp->abuf     != NULL) NCI_Free(ncp->abuf);
    if (ncp->path     != NULL) NCI_Free(ncp->path);

    double other_free_time = MPI_Wtime() - other_start;

    NCI_Free(ncp);
    if (rank == 0)
        printf("dim_free_time: %f, var_free_time: %f, other_free_time: %f\n", dim_free_time, var_free_time, other_free_time);
}

/*----< ncmpio_close_files() >-----------------------------------------------*/
int
ncmpio_close_files(NC *ncp, int doUnlink) {
    int mpireturn;

    assert(ncp != NULL); /* this should never occur */

    if (ncp->independent_fh != MPI_FILE_NULL) {
        TRACE_IO(MPI_File_close)(&ncp->independent_fh);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_error_mpi2nc(mpireturn, "MPI_File_close");
    }

    if (ncp->collective_fh != MPI_FILE_NULL) {
        TRACE_IO(MPI_File_close)(&ncp->collective_fh);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_error_mpi2nc(mpireturn, "MPI_File_close");
    }

    if (doUnlink) {
        /* called from ncmpi_abort, if the file is being created and is still
         * in define mode, the file is deleted */
        TRACE_IO(MPI_File_delete)((char *)ncp->path, ncp->mpiinfo);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_error_mpi2nc(mpireturn, "MPI_File_delete");
    }
    return NC_NOERR;
}

/*----< ncmpio_close() >------------------------------------------------------*/
/* This function is collective */
int
ncmpio_close(void *ncdp)
{
    int err=NC_NOERR, status=NC_NOERR;
    NC *ncp = (NC*)ncdp;
    double close_start = MPI_Wtime();

    if (NC_indef(ncp)) { /* currently in define mode */
        status = ncmpio__enddef(ncp, 0, 0, 0, 0); /* TODO: defaults */

        if (status != NC_NOERR) {
            /* To do: Abort new definition, if any */
            if (ncp->old != NULL) {
                ncmpio_free_NC(ncp->old);
                ncp->old = NULL;
                fClr(ncp->flags, NC_MODE_DEF);
            }
        }
    }

    if (!NC_readonly(ncp) &&  /* file is open for write */
         NC_indep(ncp)) {     /* exit independent data mode will sync header */
        err = ncmpio_end_indep_data(ncp);
        if (status == NC_NOERR) status = err;
    }

    /* if entering this function in  collective data mode, we do not have to
     * update header in file, as file header is always up-to-date */

#ifdef ENABLE_SUBFILING
    /* ncmpio__enddef() will update ncp->num_subfiles */
    /* TODO: should check ncid_sf? */
    /* if the file has subfiles, close them first */
    if (ncp->num_subfiles > 1) {
        err = ncmpio_subfile_close(ncp);
        if (status == NC_NOERR) status = err;
    }
#endif

    /* We can cancel or complete all outstanding nonblocking I/O.
     * For now, cancelling makes more sense. */
#ifdef COMPLETE_NONBLOCKING_IO
    if (ncp->numLeadGetReqs > 0) {
        err = ncmpio_wait(ncp, NC_GET_REQ_ALL, NULL, NULL, NC_REQ_INDEP);
        if (status == NC_NOERR) status = err;
        if (status == NC_NOERR) status = NC_EPENDING;
    }
    if (ncp->numLeadPutReqs > 0) {
        err = ncmpio_wait(ncp, NC_PUT_REQ_ALL, NULL, NULL, NC_REQ_INDEP);
        if (status == NC_NOERR) status = err;
        if (status == NC_NOERR) status = NC_EPENDING;
    }
#else
    if (ncp->numLeadGetReqs > 0) {
        int rank;
        MPI_Comm_rank(ncp->comm, &rank);
        printf("PnetCDF warning: %d nonblocking get requests still pending on process %d. Cancelling ...\n",ncp->numLeadGetReqs,rank);
        err = ncmpio_cancel(ncp, NC_GET_REQ_ALL, NULL, NULL);
        if (status == NC_NOERR) status = err;
        if (status == NC_NOERR) status = NC_EPENDING;
    }
    if (ncp->numLeadPutReqs > 0) {
        int rank;
        MPI_Comm_rank(ncp->comm, &rank);
        printf("PnetCDF warning: %d nonblocking put requests still pending on process %d. Cancelling ...\n",ncp->numLeadPutReqs,rank);
        err = ncmpio_cancel(ncp, NC_PUT_REQ_ALL, NULL, NULL);
        if (status == NC_NOERR) status = err;
        if (status == NC_NOERR) status = NC_EPENDING;
    }
#endif

    /* calling MPI_File_close() */
    err = ncmpio_close_files(ncp, 0);
    if (status == NC_NOERR) status = err;

    /* file is open for write and no variable has been defined */
    if (!NC_readonly(ncp) && ncp->vars.ndefined == 0) {
        int rank;

        /* wait until all processes close the file */
        MPI_Barrier(ncp->comm);

        MPI_Comm_rank(ncp->comm, &rank);
        if (rank == 0) {
            /* ignore all errors, as unexpected file size if not a fatal error */
#ifdef HAVE_TRUNCATE
            /* when calling POSIX I/O, remove file type prefix from file name */
            char *path = ncmpii_remove_file_system_type_prefix(ncp->path);
            int fd = open(path, O_RDWR, 0666);
            if (fd != -1) {
                /* obtain file size */
                off_t file_size = lseek(fd, 0, SEEK_END);
                /* truncate file size to header size, if larger than header */
                if (file_size > ncp->xsz && ftruncate(fd, ncp->xsz) < 0) {
                    err = ncmpii_error_posix2nc("ftruncate");
                    if (status == NC_NOERR) status = err;
                }
                close(fd);
            }
#else
            MPI_File fh;
            int mpireturn;
            mpireturn = MPI_File_open(MPI_COMM_SELF, ncp->path, MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
            if (mpireturn == MPI_SUCCESS) {
                /* obtain file size */
                MPI_Offset *file_size;
                MPI_File_seek(fh, 0, MPI_SEEK_END);
                MPI_File_get_position(fh, &file_size);
                /* truncate file size to header size, if larger than header */
                if (file_size > ncp->xsz) {
                    mpireturn = MPI_File_set_size(fh, ncp->xsz);
                    if (mpireturn != MPI_SUCCESS) {
                        err = ncmpii_error_mpi2nc(mpireturn,"MPI_File_set_size");
                        if (status == NC_NOERR) status = err;
                    }
                }
                MPI_File_close(&fh);
            }
            else {
                err = ncmpii_error_mpi2nc(mpireturn,"MPI_File_open");
                if (status == NC_NOERR) status = err;
            }
#endif
        }
        MPI_Barrier(ncp->comm);
    }

    /* free up space occupied by the header metadata */
    /* free up space occupied by the header metadata */
    double free_time_start = MPI_Wtime();
    // free_counter = 0;
    cls_counter = 0;
    int rank;
    MPI_Comm_rank(ncp->comm, &rank);
    ncmpio_free_NC(ncp);
    double free_time = MPI_Wtime() - free_time_start;

    if (rank == 0){
        printf("ncmpio_free_NC time: %f\n", free_time);
        printf("before ncmpio_free_NC time: %f\n", free_time_start - close_start);
        // printf("ncmpio_free_NC free() count: %d\n", free_counter);
    }
    return status;
}

