/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <pnetcdf.h>
#include <dispatch.h>
#include <pnc_debug.h>
#include <common.h>

/*----< ncmpi_def_block() >----------------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi_def_block(int         ncid,    /* IN:  file ID */
              const char *name,    /* IN:  name of block */
              int        *blkidp)  /* OUT: block ID */
{
    int err=NC_NOERR, blkid;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (!(pncp->flag & NC_MODE_DEF)) { /* must be called in define mode */
        DEBUG_ASSIGN_ERROR(err, NC_ENOTINDEFINE)
        goto err_check;
    }

    if (name == NULL || *name == 0) { /* name cannot be NULL or NULL string */
        DEBUG_ASSIGN_ERROR(err, NC_EBADNAME)
        goto err_check;
    }

    if (strlen(name) > NC_MAX_NAME) { /* name length */
        DEBUG_ASSIGN_ERROR(err, NC_EMAXNAME)
        goto err_check;
    }

    /* check if the name string is legal for the netcdf format */
    err = ncmpii_check_name(name, pncp->format);
    if (err != NC_NOERR) {
        DEBUG_TRACE_ERROR(err)
        goto err_check;
    }

    /* MPI_Offset is usually a signed value, but serial netcdf uses size_t.
     * In 1999 ISO C standard, size_t is an unsigned integer type of at least
     * 16 bit. */
    if (pncp->nblocks == NC_MAX_INT) {
        DEBUG_ASSIGN_ERROR(err, NC_EMAXBLKS)
        goto err_check;
    }

    /* check if the name string is previously used */
    err = pncp->driver->inq_blkid(pncp->ncp, name, NULL);
    if (err != NC_EBADBLK) {
        DEBUG_ASSIGN_ERROR(err, NC_EBADBLK)
        goto err_check;
    }
    else err = NC_NOERR;

err_check:
    if (pncp->flag & NC_MODE_SAFE) {
        int root_name_len, minE, rank, mpireturn;
        char *root_name=NULL;
        MPI_Offset root_size;

        /* check the error so far across processes */
        TRACE_COMM(MPI_Allreduce)(&err, &minE, 1, MPI_INT, MPI_MIN, pncp->comm);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_error_mpi2nc(mpireturn, "MPI_Allreduce");
        if (minE != NC_NOERR)
            return minE;
    }

    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_def_block() */
    err = pncp->driver->def_block(pncp->ncp, name, &blkid);
    if (err != NC_NOERR) return err;
    if (pncp->nblocks % PNC_BLOCKS_CHUNK == 0)
        pncp->blocks = NCI_Realloc(pncp->blocks,
                                 (pncp->nblocks+PNC_BLOCKS_CHUNK)*sizeof(PNC_block));
    pncp->blocks[pncp->nblocks].ndims = 0;
    pncp->blocks[pncp->nblocks].nvars = 0;
    pncp->blocks[pncp->nblocks].nrec_vars = 0;
    pncp->blocks[pncp->nblocks].vars = NULL;
    pncp->blocks[pncp->nblocks].unlimdimid = -1;

    pncp->nblocks++;//we increment the number of blocks of pncp here but it is supposed to be the total across all processes

    if (blkidp != NULL) *blkidp = blkid;

    return NC_NOERR;
}

/*----< ncmpi_inq_blkid() >--------------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_blkid(int         ncid,    /* IN:  file ID */
                const char *name,    /* IN:  name of block */
                int        *blkidp)  /* OUT: block ID */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (name == NULL || *name == 0) DEBUG_RETURN_ERROR(NC_EBADNAME)

    if (strlen(name) > NC_MAX_NAME) DEBUG_RETURN_ERROR(NC_EMAXNAME)

    /* calling the subroutine that implements ncmpi_inq_blkid() */
    return pncp->driver->inq_blkid(pncp->ncp, name, blkidp);
}

/*----< ncmpi_inq() >--------------------------------------------------------*/
int ncmpi_inq_block(int ncid,
                    int blkid,
                    char *name,
                    int *ndimsp, 
                    int *nvarsp, 
                    int *unlimdimidp)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_inq() */
    return pncp->driver->inq_block(pncp->ncp, blkid, name, ndimsp, nvarsp, unlimdimidp);
}

/*----< ncmpi_inq_blockname() >------------------------------------------------*/
/* This is an independent subroutine */
int
ncmpi_inq_blockname(int   ncid,    /* IN:  file ID */
                  int   blkid,   /* IN:  blk ID */
                  char *name)    /* OUT: name of block */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* using NC_GLOBAL in varid is illegal for this API. See
     * http://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2015/msg00196.html
     */
    if (blkid == NC_GLOBAL) DEBUG_RETURN_ERROR(NC_EGLOBAL)

    /* check whether block ID is valid */
    if (blkid < 0 || blkid >= pncp->nblocks) DEBUG_RETURN_ERROR(NC_ENOTVAR)

    /* calling the subroutine that implements ncmpi_inq_varname() */
    //META: TODO: fix blkid here by adding blkid input argument to the function 
    return pncp->driver->inq_block(pncp->ncp, blkid, name, NULL, NULL, NULL);
}