/*
 *  Copyright (C) 2018, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

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

#include "ncadios_internal.h"

int ncadiosi_inq_varid(NC_ad* ncadp, char* name, int *id) {
    int tmp;

    //return NC_NOERR;

    if (id != NULL){
        tmp = ncadiosi_var_list_find(&(ncadp->vars), name);
        if (tmp < 0){
            DEBUG_RETURN_ERROR(NC_ENOTVAR)
        }
        *id = tmp;
    }

    return NC_NOERR;
}

/*
int ncadiosi_inq_attid(NC_ad* ncadp, int vid, char* name, int *id) {
    int i;
    int attid;

    //*id = -1;

    //return NC_EINVAL;

    if (vid == NC_GLOBAL){
        attid = ncadiosi_att_list_find(&(ncadp->atts), name);
    }
    else{
        if (vid >= ncadp->vars.cnt){
            DEBUG_RETURN_ERROR(NC_EINVAL);
        }
        attid = ncadiosi_att_list_find(&(ncadp->vars.data[vid].atts), name);
    }

    if (attid < 0){
        DEBUG_RETURN_ERROR(NC_EINVAL);
    }

    if (id != NULL){
        *id = attid;
    }
    
    return NC_NOERR;
}
*/

int ncadiosi_inq_dimid(NC_ad* ncadp, char* name, int *id) {
    int tmp;

    //return NC_NOERR;

    if (id != NULL){
        tmp = ncadiosi_dim_list_find(&(ncadp->dims), name);
        if (tmp < 0){
            DEBUG_RETURN_ERROR(NC_EBADDIM)
        }
        *id = tmp;
    }

    return NC_NOERR;
}

int ncadiosi_def_var(NC_ad* ncadp, char* name, nc_type type, int ndim, int *dimids, int *id) {
    NC_ad_var var;

    //return NC_NOERR;

    if (CHECK_NAME(name)){
        var.type = type;
        var.ndim = ndim;
        var.dimids = NCI_Malloc(SIZEOF_INT * ndim);
        memcpy(var.dimids, dimids, SIZEOF_INT * ndim);
        var.name = NCI_Malloc(strlen(name) + 1);
        strcpy(var.name, name);
        ncadiosi_att_list_init(&(var.atts));
        *id = ncadiosi_var_list_add(&(ncadp->vars), var);
    }

    return NC_NOERR;
}

int ncadiosi_def_dim(NC_ad* ncadp, char* name, int len, int *id) {
    NC_ad_dim dim;
    
    //return NC_NOERR;

    if (CHECK_NAME(name)){
        dim.len = len;
        dim.name = NCI_Malloc(strlen(name) + 1);
        strcpy(dim.name, name);
    
        *id = ncadiosi_dim_list_add(&(ncadp->dims), dim);
    }

    return NC_NOERR;
}

int ncadiosi_parse_attrs(NC_ad* ncadp) {
    int i, j;
    int vid;
    char path[1024];
    char *aname = NULL;
    char *vname = NULL;


    NC_ad_var *var;
    NC_ad_dim dim;

    for (i = 0; i < ncadp->fp->nattrs; i++) {
        strcpy(path, ncadp->fp->attr_namelist[i]);

        for(j = 0; path[j] != '\0'; j++){
            if (j > 0 && path[j] == '/'){
                path[j] = '\0';
                aname = path + j + 1;
                if (path[0] == '/'){
                    vname = path + 1;
                }
                else{
                    vname = path;
                }

                vid = ncadiosi_var_list_find(&(ncadp->vars), vname);
                if (vid > -1){
                    ncadiosi_att_list_add(&(ncadp->vars.data[vid].atts), i);
                }

                break;
            }
        }
    }

    return NC_NOERR;
}

int ncadiosi_parse_rec_dim(NC_ad *ncadp) {
    int err;
    int i, j;
    
    // Find record dimension
    ncadp->recdim = -1;
    for(i = 0; i < ncadp->dims.cnt; i++){
        if (ncadp->dims.data[i].len == NC_UNLIMITED){
            ncadp->recdim = i;
            break;
        }
    }

    // Find record dimension size
    ncadp->nrec = 0;
    for(i = 0; i < ncadp->vars.cnt; i++){
        for(j = 0; j < ncadp->vars.data[i].ndim; j++){
            // Found a record variable
            if (ncadp->vars.data[i].dimids[j] == ncadp->recdim){
                ADIOS_VARINFO * v;

                // Get var info
                v = adios_inq_var(ncadp->fp, ncadp->vars.data[i].name);
                if (v == NULL){
                    err = ncmpii_error_adios2nc(adios_errno, "get_var");
                    DEBUG_RETURN_ERROR(err);
                }

                // Update record dim size
                if (ncadp->nrec < v->dims[j]){
                    ncadp->nrec = v->dims[j];
                }

                adios_free_varinfo(v);
            }
        }
    }
}