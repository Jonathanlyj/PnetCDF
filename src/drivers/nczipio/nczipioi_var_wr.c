/*
 *  Copyright (C) 2019, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the following PnetCDF APIs.
 *
 * ncmpi_get_var<kind>_all()        : dispatcher->get_var()
 * ncmpi_put_var<kind>_all()        : dispatcher->put_var()
 * ncmpi_get_var<kind>_<type>_all() : dispatcher->get_var()
 * ncmpi_put_var<kind>_<type>_all() : dispatcher->put_var()
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <mpi.h>

#include <pnc_debug.h>
#include <common.h>
#include <nczipio_driver.h>
#include "nczipio_internal.h"
#include "../ncmpio/ncmpio_NC.h"

int nczipioi_save_var(NC_zip *nczipp, NC_zip_var *varp) {
    int i, j, k, l, err;
    int *zsizes, *zsizes_all;
    MPI_Datatype mtype, ftype;  // Memory and file datatype
    int wcnt;
    int *lens;
    MPI_Aint *disps;
    MPI_Status status;
    MPI_Offset *zoffs;
    MPI_Offset start, count, oldzoff;
    void **zbufs;
    int zdimid, mdimid;
    int put_size;
    char name[128]; // Name of objects
    NC *ncp = (NC*)(nczipp->ncp);
    NC_var *ncvarp;

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO)

    // Allocate buffer for compression
    zsizes = (int*)NCI_Malloc(sizeof(int) * varp->nchunk);
    zbufs = (void**)NCI_Malloc(sizeof(void*) * varp->nmychunks);
    zsizes_all = varp->data_lens;
    zoffs = varp->data_offs;
    oldzoff = zoffs[varp->nchunk];

    // Allocate buffer for I/O
    wcnt = varp->nmychunks;
    lens = (int*)NCI_Malloc(sizeof(int) * wcnt);
    disps = (MPI_Aint*)NCI_Malloc(sizeof(MPI_Aint) * wcnt);

    memset(zsizes, 0, sizeof(int) * varp->nchunk);

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_COM)

    // Compress each chunk we own
    memset(zsizes, 0, sizeof(int) * varp->nchunk);
    for(l = 0; l < varp->nmychunks; l++){
        k = varp->mychunks[l];

        // Apply compression
        varp->zip->compress_alloc(varp->chunk_cache[k], varp->chunksize, zbufs + l, zsizes + k, varp->ndim, varp->chunkdim, varp->etype);

        // Record compressed size
        lens[l] = zsizes[k];
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_COM)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_SYNC)

    // Sync compressed data size with other processes
    CHK_ERR_ALLREDUCE(zsizes, zsizes_all, varp->nchunk, MPI_INT, MPI_MAX, nczipp->comm);
    zoffs[0] = 0;
    for(i = 0; i < varp->nchunk; i++){
        zoffs[i + 1] = zoffs[i] + zsizes_all[i];
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_SYNC)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_INIT)

    /* Write comrpessed variable
     * We start by defining data variable and writing metadata
     * Then, we create buffer type and file type for data
     * Finally MPI collective I/O is used for writing data
     */

    // Enter redefine mode
    nczipp->driver->redef(nczipp->ncp);

    // Prepare metadata variable
    if (varp->offvarid < 0 || varp->expanded){    // Check if we need new metadata vars
        // Define dimension for metadata variable
        sprintf(name, "_compressed_meta_dim_%d_%d", varp->varid, varp->metaserial);
        err = nczipp->driver->def_dim(nczipp->ncp, name, varp->nchunk + 1, &mdimid);
        if (err != NC_NOERR) return err;

        // Define off variable
        sprintf(name, "_compressed_offset_%d_%d", varp->varid, varp->metaserial);
        err = nczipp->driver->def_var(nczipp->ncp, name, NC_INT64, 1, &mdimid, &(varp->offvarid));
        if (err != NC_NOERR) return err;

        // Define lens variable
        sprintf(name, "_compressed_size_%d_%d", varp->varid, varp->metaserial);
        err = nczipp->driver->def_var(nczipp->ncp, name, NC_INT, 1, &mdimid, &(varp->lenvarid));
        if (err != NC_NOERR) return err;

        // Mark as meta variable
        i = NC_ZIP_VAR_META;
        err = nczipp->driver->put_att(nczipp->ncp, varp->offvarid, "_varkind", NC_INT, 1, &i, MPI_INT);
        if (err != NC_NOERR) return err;

        err = nczipp->driver->put_att(nczipp->ncp, varp->lenvarid, "_varkind", NC_INT, 1, &i, MPI_INT);
        if (err != NC_NOERR) return err;

        // Record lens variable id
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_lenvarid", NC_INT, 1, &(varp->lenvarid), MPI_INT);
        if (err != NC_NOERR) return err;
        
        // Record offset variable id
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_offvarid", NC_INT, 1, &varp->offvarid, MPI_INT);
        if (err != NC_NOERR) return err;

        // Record serial
        varp->metaserial++;
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_metaserial", NC_INT, 1, &(varp->metaserial), MPI_INT);
        if (err != NC_NOERR) return err;
    }

    // Prepare data variable
    if (varp->datavarid < 0|| varp->expanded || zoffs[varp->nchunk] > oldzoff){ // Check if we need new data vars
        // Define dimension for data variable
        sprintf(name, "_compressed_data_dim_%d_%d", varp->varid, varp->dataserial);
        err = nczipp->driver->def_dim(nczipp->ncp, name, zoffs[varp->nchunk], &zdimid);
        if (err != NC_NOERR) return err;

        // Define data variable
        sprintf(name, "_compressed_data_%d_%d", varp->varid, varp->dataserial);
        err = nczipp->driver->def_var(nczipp->ncp, name, NC_BYTE, 1, &zdimid, &(varp->datavarid));
        if (err != NC_NOERR) return err;

        // Record data variable id
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_datavarid", NC_INT, 1, &(varp->datavarid), MPI_INT);
        if (err != NC_NOERR) return err;

        // Mark as data variable
        i = NC_ZIP_VAR_DATA;
        err = nczipp->driver->put_att(nczipp->ncp, varp->datavarid, "_varkind", NC_INT, 1, &i, MPI_INT);
        if (err != NC_NOERR) return err;

        // Record serial
        varp->dataserial++;
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_dataserial", NC_INT, 1, &(varp->dataserial), MPI_INT);
        if (err != NC_NOERR) return err;
    }

    // unset expand flag
    varp->expanded = 0;

    // Switch to data mode
    err = nczipp->driver->enddef(nczipp->ncp);
    if (err != NC_NOERR) return err;

    // Record offset and lens
    start = 0;
    if (nczipp->rank == varp->chunk_owner[0]){
        count = varp->nchunk + 1;
    }
    else{
        count = 0;
    }
    err = nczipp->driver->put_var(nczipp->ncp, varp->offvarid, &start, &count, NULL, NULL, varp->data_offs, -1, MPI_LONG_LONG, NC_REQ_WR | NC_REQ_BLK | NC_REQ_HL | NC_REQ_COLL);
    if (err != NC_NOERR) return err;

    if (nczipp->rank == varp->chunk_owner[0]){
        count = varp->nchunk;
    }
    else{
        count = 0;
    }
    err = nczipp->driver->put_var(nczipp->ncp, varp->lenvarid, &start, &count, NULL, NULL, varp->data_lens, -1, MPI_INT, NC_REQ_WR | NC_REQ_BLK | NC_REQ_HL | NC_REQ_COLL);
    if (err != NC_NOERR) return err;

    /* Carry our coll I/O
     * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when count is 0
     * We use a dummy call inplace of type with 0 count
     */
    if (wcnt > 0){
        // Create file type
        ncvarp = ncp->vars.value[varp->datavarid];
        for(l = 0; l < varp->nmychunks; l++){
            k = varp->mychunks[l];

            // Record compressed size
            lens[l] = zsizes[k];
            disps[l] = (MPI_Aint)zoffs[k] + (MPI_Aint)ncvarp->begin;
        }
        MPI_Type_create_hindexed(wcnt, lens, disps, MPI_BYTE, &ftype);
        CHK_ERR_TYPE_COMMIT(&ftype);

        // Create memory buffer type
        for(l = 0; l < varp->nmychunks; l++){
            k = varp->mychunks[l];

            // Record compressed size
            lens[l] = zsizes[k];
            disps[l] = (MPI_Aint)zbufs[l];
        }
        err = MPI_Type_create_hindexed(wcnt, lens, disps, MPI_BYTE, &mtype);
        CHK_ERR_TYPE_COMMIT(&mtype);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_INIT)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_WR)

        // Perform MPI-IO
        // Set file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
        // Write data
        CHK_ERR_WRITE_AT_ALL(ncp->collective_fh, 0, MPI_BOTTOM, 1, mtype, &status);
        // Restore file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_WR)

#ifdef _USE_MPI_GET_COUNT
        MPI_Get_count(&status, MPI_BYTE, &put_size);
#else
        MPI_Type_size(mtype, &put_size);
#endif
        nczipp->putsize += put_size;

        // Free type
        MPI_Type_free(&ftype);
        MPI_Type_free(&mtype);
    }
    else{
        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_INIT)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_WR)

        // Follow coll I/O with dummy call
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
        CHK_ERR_WRITE_AT_ALL(ncp->collective_fh, 0, MPI_BOTTOM, 0, MPI_BYTE, &status);
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_WR)
    }

    // Free buffers
    NCI_Free(zsizes);
    for(l = 0; l < varp->nmychunks; l++){
        NCI_Free(zbufs[l]);
    }
    NCI_Free(zbufs);

    NCI_Free(lens);
    NCI_Free(disps);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO)

    return NC_NOERR;
}

int nczipioi_save_nvar(NC_zip *nczipp, int nvar, int *varids) {
    NC_zip_var *varp;
    int i, j, k, l, err;
    int vid;    // Iterator for variable id
    int cid;    // Iterator for chunk id
    int max_nchunks = 0;
    int *zsizes, *zsizes_all;
    MPI_Offset *zoffs;
    MPI_Offset start, count, oldzoff;
    MPI_Datatype mtype, ftype;  // Memory and file datatype
    int wcnt, wcur;
    int *mlens, *flens;
    MPI_Aint *mdisps, *fdisps;
    MPI_Status status;
    int put_size;
    void **zbufs;
    int zdimid, mdimid;
    char name[128]; // Name of objects
    NC *ncp = (NC*)(nczipp->ncp);
    NC_var *ncvarp;

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_INIT)

    wcnt = 0;
    for(i = 0; i < nvar; i++){
        vid = varids[i];
        wcnt += nczipp->vars.data[vid].nmychunks;
        if (max_nchunks < nczipp->vars.data[vid].nchunk){
            max_nchunks = nczipp->vars.data[vid].nchunk;
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_INIT)

    // Allocate buffer for compression
    zsizes = (int*)NCI_Malloc(sizeof(int) * max_nchunks);
    zbufs = (void**)NCI_Malloc(sizeof(void*) * wcnt);

    // Allocate buffer file type
    mlens = (int*)NCI_Malloc(sizeof(int) * wcnt);
    mdisps = (MPI_Aint*)NCI_Malloc(sizeof(MPI_Aint) * wcnt);
    flens = (int*)NCI_Malloc(sizeof(int) * wcnt);
    fdisps = (MPI_Aint*)NCI_Malloc(sizeof(MPI_Aint) * wcnt);

    // Enter redefine mode
    nczipp->driver->redef(nczipp->ncp);

    wcur = 0;
    for(vid = 0; vid < nvar; vid++){
        varp = nczipp->vars.data + varids[vid];

        NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_COM)

        zsizes_all = varp->data_lens;
        zoffs = varp->data_offs;
        oldzoff = zoffs[varp->nchunk];

        memset(zsizes, 0, sizeof(int) * varp->nchunk);

        // Compress each chunk we own
        memset(zsizes, 0, sizeof(int) * varp->nchunk);
        for(l = 0; l < varp->nmychunks; l++){
            cid = varp->mychunks[l];

            // Apply compression
            varp->zip->compress_alloc(varp->chunk_cache[cid], varp->chunksize, zbufs + wcur + l, zsizes + cid, varp->ndim, varp->chunkdim, varp->etype);
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_COM)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_SYNC)

        // Sync compressed data size with other processes
        CHK_ERR_ALLREDUCE(zsizes, zsizes_all, varp->nchunk, MPI_INT, MPI_MAX, nczipp->comm);
        zoffs[0] = 0;
        for(cid = 0; cid < varp->nchunk; cid++){
            zoffs[cid + 1] = zoffs[cid] + zsizes_all[cid];
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_SYNC)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_INIT)

        /* Write comrpessed variable
        * We start by defining data variable and writing metadata
        * Then, we create buffer type and file type for data
        * Finally MPI collective I/O is used for writing data
        */

        // Prepare metadata variable
        if (varp->offvarid < 0 || varp->expanded){    // Check if we need new metadata vars
            // Define dimension for metadata variable
            sprintf(name, "_compressed_meta_dim_%d_%d", varp->varid, varp->metaserial);
            err = nczipp->driver->def_dim(nczipp->ncp, name, varp->nchunk + 1, &mdimid);
            if (err != NC_NOERR) return err;

            // Define off variable
            sprintf(name, "_compressed_offset_%d_%d", varp->varid, varp->metaserial);
            err = nczipp->driver->def_var(nczipp->ncp, name, NC_INT64, 1, &mdimid, &(varp->offvarid));
            if (err != NC_NOERR) return err;

            // Define lens variable
            sprintf(name, "_compressed_size_%d_%d", varp->varid, varp->metaserial);
            err = nczipp->driver->def_var(nczipp->ncp, name, NC_INT, 1, &mdimid, &(varp->lenvarid));
            if (err != NC_NOERR) return err;

            // Mark as meta variable
            i = NC_ZIP_VAR_META;
            err = nczipp->driver->put_att(nczipp->ncp, varp->offvarid, "_varkind", NC_INT, 1, &i, MPI_INT);
            if (err != NC_NOERR) return err;

            err = nczipp->driver->put_att(nczipp->ncp, varp->lenvarid, "_varkind", NC_INT, 1, &i, MPI_INT);
            if (err != NC_NOERR) return err;

            // Record lens variable id
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_lenvarid", NC_INT, 1, &(varp->lenvarid), MPI_INT);
            if (err != NC_NOERR) return err;
            
            // Record offset variable id
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_offvarid", NC_INT, 1, &varp->offvarid, MPI_INT);
            if (err != NC_NOERR) return err;

            // Record serial
            varp->metaserial++;
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_metaserial", NC_INT, 1, &(varp->metaserial), MPI_INT);
            if (err != NC_NOERR) return err;
        }

        // Prepare data variable
        if (varp->datavarid < 0|| varp->expanded || zoffs[varp->nchunk] > oldzoff){ // Check if we need new data vars
            // Define dimension for data variable
            sprintf(name, "_compressed_data_dim_%d_%d", varp->varid, varp->dataserial);
            err = nczipp->driver->def_dim(nczipp->ncp, name, zoffs[varp->nchunk], &zdimid);
            if (err != NC_NOERR) return err;

            // Define data variable
            sprintf(name, "_compressed_data_%d_%d", varp->varid, varp->dataserial);
            err = nczipp->driver->def_var(nczipp->ncp, name, NC_BYTE, 1, &zdimid, &(varp->datavarid));
            if (err != NC_NOERR) return err;

            // Record data variable id
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_datavarid", NC_INT, 1, &(varp->datavarid), MPI_INT);
            if (err != NC_NOERR) return err;

            // Mark as data variable
            i = NC_ZIP_VAR_DATA;
            err = nczipp->driver->put_att(nczipp->ncp, varp->datavarid, "_varkind", NC_INT, 1, &i, MPI_INT);
            if (err != NC_NOERR) return err;

            // Record serial
            varp->dataserial++;
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_dataserial", NC_INT, 1, &(varp->dataserial), MPI_INT);
            if (err != NC_NOERR) return err;
        }

        // unset expand flag
        varp->expanded = 0;

        /* Paramemter for file and memory type 
         * We do not know variable file offset until the end of define mode
         * We will add the displacement later
         */
        for(l = 0; l < varp->nmychunks; l++){
            cid = varp->mychunks[l];

            // Record parameter
            flens[wcur + l] = zsizes[cid];
            fdisps[wcur + l] = (MPI_Aint)zoffs[cid];
            mlens[wcur + l] = zsizes[cid];
            mdisps[wcur + l] = (MPI_Aint)zbufs[wcur + l];
        }

        // Move to parameters for next variable
        wcur += varp->nmychunks;

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_INIT)
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_INIT)

    // Switch back to data mode
    err = nczipp->driver->enddef(nczipp->ncp);
    if (err != NC_NOERR) return err;

    // Record offset and lens
    for(vid = 0; vid < nvar; vid++){
        varp = nczipp->vars.data + varids[vid];
        start = 0;
        if (nczipp->rank == varp->chunk_owner[0]){
            count = varp->nchunk + 1;
        }
        else{
            count = 0;
        }
        err = nczipp->driver->put_var(nczipp->ncp, varp->offvarid, &start, &count, NULL, NULL, varp->data_offs, -1, MPI_LONG_LONG, NC_REQ_WR | NC_REQ_BLK | NC_REQ_HL | NC_REQ_COLL);
        if (err != NC_NOERR) return err;

        if (nczipp->rank == varp->chunk_owner[0]){
            count = varp->nchunk;
        }
        else{
            count = 0;
        }
        err = nczipp->driver->put_var(nczipp->ncp, varp->lenvarid, &start, &count, NULL, NULL, varp->data_lens, -1, MPI_INT, NC_REQ_WR | NC_REQ_BLK | NC_REQ_HL | NC_REQ_COLL);
        if (err != NC_NOERR) return err;
    }

    /* Now it's time to add variable file offset to displacements
     * File type offset need to be specified in non-decreasing order
     * We assume ncmpio place variable according to the order they are declared
     */
    wcur = 0;
    for(vid = 0; vid < nvar; vid++){
        varp = nczipp->vars.data +  varids[vid];
        ncvarp = ncp->vars.value[varp->datavarid];
        for(l = 0; l < varp->nmychunks; l++){
            cid = varp->mychunks[l];
            // Adjust file displacement
            fdisps[wcur++] += (MPI_Aint)ncvarp->begin;
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_INIT)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_IO_WR)

    /* Carry our coll I/O
     * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when count is 0
     * We use a dummy call inplace of type with 0 count
     */
    if (wcnt > 0){
         // Create file type
        MPI_Type_create_hindexed(wcnt, flens, fdisps, MPI_BYTE, &ftype);
        CHK_ERR_TYPE_COMMIT(&ftype);

        // Create memmory type
        MPI_Type_create_hindexed(wcnt, mlens, mdisps, MPI_BYTE, &mtype);
        CHK_ERR_TYPE_COMMIT(&mtype);

        // Perform MPI-IO
        // Set file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
        // Write data
        CHK_ERR_WRITE_AT_ALL(ncp->collective_fh, 0, MPI_BOTTOM, 1, mtype, &status);
        // Restore file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_WR)

#ifdef _USE_MPI_GET_COUNT
        MPI_Get_count(&status, MPI_BYTE, &put_size);
#else
        MPI_Type_size(mtype, &put_size);
#endif
        nczipp->putsize += put_size;

        // Free type
        MPI_Type_free(&ftype);
        MPI_Type_free(&mtype);
    }
    else{
        // Follow coll I/O with dummy call
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
        CHK_ERR_WRITE_AT_ALL(ncp->collective_fh, 0, MPI_BOTTOM, 0, MPI_BYTE, &status);
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO_WR)
    }

    // Free buffers
    NCI_Free(zsizes);
    for(l = 0; l < varp->nmychunks; l++){
        NCI_Free(zbufs[l]);
    }
    NCI_Free(zbufs);

    NCI_Free(flens);
    NCI_Free(fdisps);
    NCI_Free(mlens);
    NCI_Free(mdisps);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_IO)

    return NC_NOERR;
}