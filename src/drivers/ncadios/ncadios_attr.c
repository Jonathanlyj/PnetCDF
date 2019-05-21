/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the following PnetCDF APIs.
 *
 * ncmpi_inq_attname() : dispatcher->inq_attname()
 * ncmpi_inq_attid()   : dispatcher->inq_attid()
 * ncmpi_inq_att()     : dispatcher->inq_att()
 * ncmpi_rename_att()  : dispatcher->inq_rename_att()
 * ncmpi_copy_att()    : dispatcher->inq_copy_att()
 * ncmpi_del_att()     : dispatcher->inq_del_att()
 * ncmpi_get_att()     : dispatcher->inq_get_att()
 * ncmpi_put_att()     : dispatcher->inq_put_arr()
 *
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <pnc_debug.h>
#include <common.h>
#include <ncadios_driver.h>


int
ncadios_inq_attname(void *ncdp,
                  int   varid,
                  int   attid,
                  char *name)
{
    int err;
    NC_ad *ncadp = (NC_ad*)ncdp;

    /* Only global attr is defined */
    /*
    if (varid >= 0){
        DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);
    }
    */

    if (attid >= ncadp->fp->nattrs){
        DEBUG_RETURN_ERROR(NC_EINVAL);
    }

    if (name != NULL){
        strcpy(name, ncadp->fp->attr_namelist[attid]);
    }

    return NC_NOERR;
}

int
ncadios_inq_attid(void       *ncdp,
                int         varid,
                const char *name,
                int        *attidp)
{
    int err;
    int i;
    NC_ad *ncadp = (NC_ad*)ncdp;

    for(i = 0; i < ncadp->fp->nattrs; i++){
        if (strcmp(name, ncadp->fp->attr_namelist[i]) == 0){
            if (attidp != NULL){
                *attidp = i;
            }
            break;
        }
    }

    // Name not found
    if (i >= ncadp->fp->nattrs){
        DEBUG_RETURN_ERROR(NC_EINVAL);
    }

    return NC_NOERR;
}

int
ncadios_inq_att(void       *ncdp,
              int         varid,
              const char *name,
              nc_type    *datatypep,
              MPI_Offset *lenp)
{
    int err;
    NC_ad *ncadp = (NC_ad*)ncdp;
    enum ADIOS_DATATYPES atype;
    int  asize, tsize;
    void *adata;
    
    err = adios_get_attr(ncadp->fp, name, &atype, &asize, &adata);
    if (err != 0){
        //TODO: translate adios error
        return err;
    }

    tsize = adios_type_size(atype, adata);

    if (datatypep != NULL){
        *datatypep = ncadios_to_nc_type(atype);
    }

    if (lenp != NULL){
        if (atype == adios_string){
            *lenp = (MPI_Offset)asize;
        }
        else{
            *lenp = (MPI_Offset)asize / tsize;
        }
    }

    return NC_NOERR;
}

int
ncadios_rename_att(void       *ncdp,
                 int         varid,
                 const char *name,
                 const char *newname)
{
    int err;
    NC_ad *ncadp = (NC_ad*)ncdp;

    /* Read only driver */
    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    return NC_NOERR;
}


int
ncadios_copy_att(void       *ncdp_in,
               int         varid_in,
               const char *name,
               void       *ncdp_out,
               int         varid_out)
{
    int err;
    NC_ad *ncadp_in  = (NC_ad*)ncdp_in;
    NC_ad *ncadp_out = (NC_ad*)ncdp_out;

    /* Read only driver */
    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    return NC_NOERR;
}

int
ncadios_del_att(void       *ncdp,
              int         varid,
              const char *name)
{
    int err;
    NC_ad *ncadp = (NC_ad*)ncdp;

    /* Read only driver */
    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    return NC_NOERR;
}

int
ncadios_get_att(void         *ncdp,
              int           varid,
              const char   *name,
              void         *buf,
              MPI_Datatype  itype)
{
    int err;
    NC_ad *ncadp = (NC_ad*)ncdp;
    enum ADIOS_DATATYPES atype;
    int  asize;
    void *adata;
    
    err = adios_get_attr(ncadp->fp, name, &atype, &asize, &adata);
    if (err != 0){
        //TODO: translate adios error
        return err;
    }

    //TODO: Convert adios type
    memcpy(buf, adata, asize);

    return NC_NOERR;
}

int
ncadios_put_att(void         *ncdp,
              int           varid,
              const char   *name,
              nc_type       xtype,
              MPI_Offset    nelems,
              const void   *buf,
              MPI_Datatype  itype)
{
    int err;
    NC_ad *ncadp = (NC_ad*)ncdp;

    /* Read only driver */
    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    return NC_NOERR;
}
