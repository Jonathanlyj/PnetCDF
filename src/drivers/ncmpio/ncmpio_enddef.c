/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the corresponding APIs defined in src/dispatchers/file.c
 *
 * ncmpi_enddef()  : dispatcher->enddef()
 * ncmpi__enddef() : dispatcher->_enddef()
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>  /* strtol() */
#include <string.h>  /* memset() */
#include <assert.h>
#include <errno.h>
#include <ctype.h>
#include <mpi.h>

#include <pnc_debug.h>
#include <common.h>
#include <ncx.h>
#include "ncmpio_NC.h"
#ifdef ENABLE_SUBFILING
#include "ncmpio_subfile.h"
#endif


/*----< move_file_block() >--------------------------------------------------*/
static int
move_file_block(NC         *ncp,
                MPI_Offset  to,     /* destination file starting offset */
                MPI_Offset  from,   /* source file starting offset */
                MPI_Offset  nbytes) /* amount to be moved */
{
    int rank, nprocs, bufcount, mpireturn, err, status=NC_NOERR, min_st;
    void *buf;
    size_t chunk_size;
    MPI_Status mpistatus;

    MPI_Comm_size(ncp->comm, &nprocs);
    MPI_Comm_rank(ncp->comm, &rank);

    /* Divide amount nbytes among all processes. If the divided amount,
     * chunk_size, is larger then MOVE_UNIT, set chunk_size to be the move unit
     * size per process (make sure it is <= NC_MAX_INT, as MPI read/write APIs
     * use 4-byte int in their count argument.)
     */
#define MOVE_UNIT 67108864
    chunk_size = nbytes / nprocs;
    if (nbytes % nprocs) chunk_size++;
    if (chunk_size > MOVE_UNIT) {
        /* move data in multiple rounds, MOVE_UNIT per process at a time */
        chunk_size = MOVE_UNIT;
    }

    /* buf will be used as a temporal buffer to move data in chunks, i.e.
     * read a chunk and later write to the new location */
    buf = NCI_Malloc(chunk_size);
    if (buf == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

    /* make fileview entire file visible */
    TRACE_IO(MPI_File_set_view)(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE,
                                "native", MPI_INFO_NULL);

    /* move the variable starting from its tail toward its beginning */
    while (nbytes > 0) {
        int get_size=0;

        /* calculate how much to move at each time. chunk_size has been
         * checked, must be < NC_MAX_INT
         */
        bufcount = (int)chunk_size;
        if (nbytes < (MPI_Offset)nprocs * chunk_size) {
            /* handle the last group of chunks */
            MPI_Offset rem_chunks = nbytes / chunk_size;
            if (rank > rem_chunks) /* these processes do not read/write */
                bufcount = 0;
            else if (rank == rem_chunks) /* this process reads/writes less */
                /* make bufcount < chunk_size */
                bufcount = (int)(nbytes % chunk_size);
            nbytes = 0;
        }
        else {
            nbytes -= chunk_size*nprocs;
        }

        /* explicitly initialize mpistatus object to 0. For zero-length read,
         * MPI_Get_count may report incorrect result for some MPICH version,
         * due to the uninitialized MPI_Status object passed to MPI-IO calls.
         * Thus we initialize it above to work around.
         */
        memset(&mpistatus, 0, sizeof(MPI_Status));

        /* read the original data @ from+nbytes+rank*chunk_size */
        TRACE_IO(MPI_File_read_at_all)(ncp->collective_fh,
                                       from+nbytes+rank*chunk_size,
                                       buf, bufcount, MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_error_mpi2nc(mpireturn, "MPI_File_read_at_all");
            if (err == NC_EFILE) DEBUG_ASSIGN_ERROR(status, NC_EREAD)
            get_size = bufcount;
        }
        else {
            /* for zero-length read, MPI_Get_count may report incorrect result
             * for some MPICH version, due to the uninitialized MPI_Status
             * object passed to MPI-IO calls. Thus we initialize it above to
             * work around. See MPICH ticket:
             * https://trac.mpich.org/projects/mpich/ticket/2332
             *
             * Note we cannot set bufcount to get_size, as the actual size
             * read from a file may be less than bufcount. Because we are
             * moving whatever read to a new file offset, we must use the
             * amount actually read to call MPI_File_write_at_all below.
             *
             * Update the number of bytes read since file open.
             * Because each rank reads and writes no more than one chunk_size
             * at a time and chunk_size is < NC_MAX_INT, it is OK to call
             * MPI_Get_count, instead of MPI_Get_count_c.
             */
            MPI_Get_count(&mpistatus, MPI_BYTE, &get_size);
            ncp->get_size += get_size;
        }

        /* MPI_Barrier(ncp->comm); */
        /* important, in case new region overlaps old region */
        TRACE_COMM(MPI_Allreduce)(&status, &min_st, 1, MPI_INT, MPI_MIN,
                                  ncp->comm);
        status = min_st;
        if (status != NC_NOERR) break;

        /* write to new location @ to+nbytes+rank*chunk_size
         *
         * Ideally, we should write the amount of get_size returned from a call
         * to MPI_Get_count in the below MPI write. This is in case some
         * variables are defined but never been written. The value returned by
         * MPI_Get_count is supposed to be the actual amount read by the MPI
         * read call. If partial data (or none) is available for read, then we
         * should just write that amount. Note this MPI write is collective,
         * and thus all processes must participate the call even if get_size
         * is 0. However, in some MPICH versions MPI_Get_count fails to report
         * the correct value due to an internal error that fails to initialize
         * the MPI_Status object. Therefore, the solution can be either to
         * explicitly initialize the status object to zeros, or to just use
         * bufcount for write. Note that the latter will write the variables
         * that have not been written before. Below uses the former option.
         */

        /* explicitly initialize mpistatus object to 0. For zero-length read,
         * MPI_Get_count may report incorrect result for some MPICH version,
         * due to the uninitialized MPI_Status object passed to MPI-IO calls.
         * Thus we initialize it above to work around.
         */
        memset(&mpistatus, 0, sizeof(MPI_Status));

        TRACE_IO(MPI_File_write_at_all)(ncp->collective_fh,
                                        to+nbytes+rank*chunk_size,
                                        buf, get_size /* NOT bufcount */,
                                        MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_error_mpi2nc(mpireturn, "MPI_File_write_at_all");
            if (err == NC_EFILE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
        }
        else {
            /* update the number of bytes written since file open.
             * Because each rank reads and writes no more than one chunk_size
             * at a time and chunk_size is < NC_MAX_INT, it is OK to call
             * MPI_Get_count, instead of MPI_Get_count_c.
             */
            int put_size;
            mpireturn = MPI_Get_count(&mpistatus, MPI_BYTE, &put_size);
            if (mpireturn != MPI_SUCCESS || put_size == MPI_UNDEFINED)
                ncp->put_size += get_size; /* or bufcount */
            else
                ncp->put_size += put_size;
        }
        TRACE_COMM(MPI_Allreduce)(&status, &min_st, 1, MPI_INT, MPI_MIN, ncp->comm);
        status = min_st;
        if (status != NC_NOERR) break;
    }
    NCI_Free(buf);
    return status;
}

/*----< move_fixed_vars() >--------------------------------------------------*/
/* move one fixed variable at a time, only when the new begin > old begin */
static int
move_fixed_vars(NC *ncp, NC *old)
{
    int i, err, status=NC_NOERR;

    /* move starting from the last fixed variable */
    for (i=old->vars.ndefined-1; i>=0; i--) {
        if (IS_RECVAR(old->vars.value[i])) continue;

        MPI_Offset from = old->vars.value[i]->begin;
        MPI_Offset to   = ncp->vars.value[i]->begin;
        if (to > from) {
            err = move_file_block(ncp, to, from, ncp->vars.value[i]->len);
            if (status == NC_NOERR) status = err;
        }
    }
    return status;
}

/*----< move_record_vars() >-------------------------------------------------*/
/* Move the record variables from lower offsets (old) to higher offsets. */
static int
move_record_vars(NC *ncp, NC *old) {
    int err;
    MPI_Offset recno;
    MPI_Offset nrecs = ncp->numrecs;
    MPI_Offset ncp_recsize = ncp->recsize;
    MPI_Offset old_recsize = old->recsize;
    MPI_Offset ncp_off = ncp->begin_rec;
    MPI_Offset old_off = old->begin_rec;

    assert(ncp_recsize >= old_recsize);

    if (ncp_recsize == old_recsize) {
        if (ncp_recsize == 0) /* no record variable defined yet */
            return NC_NOERR;

        /* No new record variable inserted, move the entire record variables
         * as a whole */
        err = move_file_block(ncp, ncp_off, old_off, ncp_recsize * nrecs);
        if (err != NC_NOERR) return err;
    } else {
        /* new record variables inserted, move one whole record at a time */
        for (recno = nrecs-1; recno >= 0; recno--) {
            err = move_file_block(ncp, ncp_off+recno*ncp_recsize,
                                       old_off+recno*old_recsize, old_recsize);
            if (err != NC_NOERR) return err;
        }
    }
    return NC_NOERR;
}
/*--------------< hdr_put_NC_modified_blockarray() >----------------------------------------------*/
//META: this is just for MPI comm at enddef
static int
hdr_put_NC_modified_blockarray(const NC_blockarray *ncpb, bufferinfo *pbp) {
    /* netCDF file format:
     *  ...
     * block_offset list     = ABSENT | NC_BLOCK nelems [block_info ...]
     * block_info            = name OFFSET bsize
     * ABSENT       = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *                ZERO  ZERO64  // for CDF-5
     * ZERO         = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64       = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_DIMENSION = \x00 \x00 \x00 \x0A         // tag for list of dimensions
     * nelems       = NON_NEG       // number of elements in following sequence
     * NON_NEG      = <non-negative INT> |        // CDF-1 and CDF-2
     *                <non-negative INT64>        // CDF-5
     */
    int i, status, n_modified_blocks = 0;


    for (i=0; i<ncpb->ndefined; i++) {
        if (ncpb->value[i]->modified) {
            n_modified_blocks++;
        }
    }
    assert(pbp != NULL);

    if (ncpb == NULL || n_modified_blocks == 0) { /* ABSENT */
        status = ncmpix_put_uint32((void**)(&pbp->pos), NC_UNSPECIFIED);
        if (status != NC_NOERR) return status;

        /* put a ZERO or ZERO64 depending on which CDF format */
        status = ncmpix_put_uint32((void**)(&pbp->pos), 0);
        if (status != NC_NOERR) return status;
    }
    else {
        /* copy NC_BLOCK */
        status = ncmpix_put_uint32((void**)(&pbp->pos), NC_BLOCK);
        if (status != NC_NOERR) return status;

        /* copy nelems */

        status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)n_modified_blocks);
        if (status != NC_NOERR) return status;

        /* copy name OFFSET block_size*/
    /* copy [dimid ...] */
    for (i=0; i<ncpb->ndefined; i++) {
        if (ncpb->value[i]->modified) {
            //copy block id

            status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)i);
            if (status != NC_NOERR) return status;
            //copy block name
            // printf("\npbp->pos - pbp->base: %ld\n", pbp->pos - pbp->base);
            status = hdr_put_NC_name(pbp, ncpb->value[i]->name);
            //copy block size
            status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)ncpb->value[i]->xsz);
            if (status != NC_NOERR) return status;
            //copy block var sie
            status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)ncpb->value[i]->block_var_len);
            if (status != NC_NOERR) return status;
            //copy block rec var sie
            status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)ncpb->value[i]->block_recvar_len);
            if (status != NC_NOERR) return status;
            }
        }
    }

    return NC_NOERR;
}
static int serialize_bufferinfo_array(NC *ncp, void *buf){
    bufferinfo sendbuff;
    int err;
    sendbuff.pos           = buf;
    sendbuff.base          = buf;
    sendbuff.version       = ncp->format;
    err = hdr_put_NC_modified_blockarray(&ncp->blocks, &sendbuff);
    if (err != NC_NOERR) {
        DEBUG_RETURN_ERROR(err)
    }
    return err;
}

static MPI_Offset
hdr_len_NC_modified_blockarray(const NC_blockarray *ncpb) {
    MPI_Offset buffer_size = 0;
    int n_modified_blocks = 0;

    // Count the number of modified blocks
    for (int i = 0; i < ncpb->ndefined; i++) {
        if (ncpb->value[i]->modified) {
            n_modified_blocks++;
        }
    }

    if (ncpb == NULL || n_modified_blocks == 0) {
        // If no modified blocks, size is for ABSENT case
        buffer_size += sizeof(uint32_t); // NC_UNSPECIFIED
        buffer_size += sizeof(uint32_t); // ZERO

    } else {
        // Size for NC_BLOCK and number of elements
        buffer_size += sizeof(uint32_t); // NC_BLOCK
        buffer_size += sizeof(uint32_t); // number of elements

        // Size for each modified block
        for (int i = 0; i < ncpb->ndefined; i++) {
            if (ncpb->value[i]->modified) {
                buffer_size += sizeof(uint32_t); // block ID

                buffer_size += sizeof(uint32_t) + _RNDUP(ncpb->value[i]->name_len, X_ALIGN); 

                buffer_size += sizeof(uint32_t); // block size

                buffer_size += sizeof(uint32_t); // block non-record var size

                buffer_size += sizeof(uint32_t); // block record var size
            }
        }
    }

    return buffer_size;
}


static int deserialize_bufferinfo_array(NC *ncp, void *buf, int *recv_displs, int *new_block_offsets, int total_size, int nproc, int rank){
    bufferinfo recvbuff;
    int err;
    recvbuff.pos           = buf;
    recvbuff.base          = buf;
    recvbuff.chunk = _RNDUP( MAX(MIN_NC_XSZ+4, ncp->chunk), X_ALIGN );


    
    for (int i=0; i<nproc; i++){
        recvbuff.pos = recvbuff.base + recv_displs[i];
        if (i < nproc - 1){
            // printf("\n nrpoc: %d\n, i: %d", nproc, i);
            recvbuff.end = recvbuff.base + recv_displs[i + 1];
        }else{
            recvbuff.end = recvbuff.base + total_size;
        }
        if (i != rank){
            //only new blockarrays need to be used to update the blockarray
            //don't overwrite the existing local block - otherwise the block content is lost
            err = hdr_get_NC_modified_blockarray(&recvbuff, &ncp->blocks, i, new_block_offsets);
            if (err != NC_NOERR) {
                return err;
            }
        }
        // err = hdr_get_NC_modified_blockarray(&recvbuff, &ncp->blocks, i, new_block_offsets);
    }
    return NC_NOERR;

}
/*--------------< hdr_get_NC_modified_blockarray() >----------------------------------------------*/
int
hdr_get_NC_modified_blockarray(bufferinfo *pbp, NC_blockarray *ncpb, int src_rank, int *new_block_offsets) {
    int i, status;
    uint32_t nelems;
    uint64_t nelems64;

    assert(pbp != NULL);
    assert(ncpb != NULL);

    /* Read the NC_BLOCK or ABSENT indicator */
    uint32_t indicator;
    status = ncmpix_get_uint32((const void**)(&pbp->pos), &indicator);
    if (status != NC_NOERR) return status;

    if (indicator == NC_UNSPECIFIED) {
        /* Read ZERO or ZERO64 */
        uint32_t zero;
        status = ncmpix_get_uint32((const void**)(&pbp->pos), &zero);
        if (status != NC_NOERR) return status;
        return NC_NOERR;
    }

    /* Read the number of elements */

    status = ncmpix_get_uint32((const void**)(&pbp->pos), &nelems);
    if (status != NC_NOERR) return status;



    /* Allocate memory for the blocks */
    // ncp->blocks = (NC_blockarray *)malloc(sizeof(NC_blockarray));
    // ncpb->ndefined = nelems;
    // ncpb->value = (NC_block **)malloc(nelems * sizeof(NC_block*));

    /* Read each block */
    int block_index;
    for (int i = 0;i < nelems;i++) {
        // Read block id
        uint32_t block_id;
        status = ncmpix_get_uint32((const void**)(&pbp->pos), &block_id);
        if (status != NC_NOERR) return status;
        if (block_id < ncpb->nread){
            block_index = (int)block_id;
            if (ncpb->value[block_index]->modified) {
                //Two process modified the same block, should error our
                return NC_EINVAL;
            }
        }else{
            block_index = (int)block_id + new_block_offsets[src_rank];
        }

        // Read block name
        char *block_name;
        size_t block_name_len;
        status = hdr_get_NC_name(pbp, &block_name, &block_name_len);
        if (status != NC_NOERR) return status;
        NC_block *blockp = (NC_block *)NCI_Malloc(sizeof(NC_block));
        blockp->vars.ndefined = 0;
        blockp->vars.value = NULL;
        blockp->vars.nameT = NULL;
        blockp->dims.ndefined = 0;
        blockp->dims.value = NULL;
        blockp->dims.nameT = NULL;
        blockp->name = block_name;
        blockp->name_len = block_name_len;

        // Read block size
        uint32_t block_size;
        status = ncmpix_get_uint32((const void**)(&pbp->pos), &block_size);
        if (status != NC_NOERR) return status;
        blockp->xsz = block_size;
        // Read block var size
        uint32_t block_var_xsz;
        status = ncmpix_get_uint32((const void**)(&pbp->pos), &block_var_xsz);
        if (status != NC_NOERR) return status;

        // Read block var size
        uint32_t block_recvar_xsz;
        status = ncmpix_get_uint32((const void**)(&pbp->pos), &block_recvar_xsz);
        if (status != NC_NOERR) return status;
    
        blockp->block_var_len = block_var_xsz;
        blockp->block_recvar_len = block_recvar_xsz;
        // Modified blocks from other processes, not by the current process
        blockp->modified = 0;  
        
        ncpb->value[block_index] = blockp;
        
    }

    return NC_NOERR;
}
/*----< NC_begins() >--------------------------------------------------------*/
/*
 * This function is only called at enddef().
 * It computes each variable's 'begin' offset, and sets/updates the followings:
 *    ncp->xsz                   ---- header size
 *    ncp->vars.value[*]->begin  ---- each variable's 'begin' offset
 *    ncp->begin_var             ---- offset of first non-record variable
 *    ncp->begin_rec             ---- offset of first     record variable
 *    ncp->recsize               ---- sum of single records
 *    ncp->numrecs               ---- number of records (set only if new file)
 */
static int
NC_begins(NC *ncp)
{
    int i, j, rank, nproc, mpireturn;
    MPI_Offset end_var = 0;
    MPI_Offset global_header_wlen, local_header_wlen;
    MPI_Offset block_var_len = 0, block_start_var = 0, block_end_var = 0, block_var_tot_len = 0; //META: for non-record var begins
    MPI_Offset block_rec_var_len = 0, block_start_rec_var = 0, block_end_rec_var = 0, block_rec_var_tot_len = 0; //META: for record var begins

    NC_var *first_var = NULL;       /* first "non-record" var */
    int err;

    /* For CDF-1 and 2 formats, a variable's "begin" in the header is 4 bytes.
     * For CDF-5, it is 8 bytes.
     */
    
    /* get the true header size (not header extent) */
    MPI_Comm_rank(ncp->comm, &rank);
    MPI_Comm_size(ncp->comm, &nproc);
    ncp->global_xsz = ncmpio_global_hdr_len_NC(ncp);
    global_header_wlen = _RNDUP(ncp->global_xsz, X_ALIGN);
    local_header_wlen = 0;
    //META: get the local header size
    for(int i=0;i<ncp->blocks.ndefined;i++){
        local_header_wlen += _RNDUP(ncp->blocks.value[i]->xsz, X_ALIGN);
        if(i==0){
            ncp->blocks.value[i]->begin = global_header_wlen;
        }else{
            ncp->blocks.value[i]->begin = _RNDUP(ncp->blocks.value[i-1]->xsz, X_ALIGN) + ncp->blocks.value[i-1]->begin;
        }
    }
    ncp->xsz = global_header_wlen + _RNDUP(local_header_wlen, X_ALIGN);
    //META: get the total header size

    // ncp->blocks.value[0]->xsz = ncmpio_local_hdr_len_NC(ncp->blocks.value[0]);
    // if (rank == 1) printf("\nrank %d, pncp->ncp->global_xsz: %lld", rank, ncp->global_xsz);

    
    if (ncp->safe_mode) { /* this consistency check is redundant as metadata is
                             kept consistent at all time when safe mode is on */
        int err, status;
        MPI_Offset root_xsz = ncp->xsz;

        /* only root's header size matters */
        TRACE_COMM(MPI_Bcast)(&root_xsz, 1, MPI_OFFSET, 0, ncp->comm);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_error_mpi2nc(mpireturn, "MPI_Bcast");
            DEBUG_RETURN_ERROR(err)
        }
        err = NC_NOERR;
        if (root_xsz != ncp->xsz) DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE)

        /* find min error code across processes */
        TRACE_COMM(MPI_Allreduce)(&err, &status, 1, MPI_INT, MPI_MIN,ncp->comm);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_error_mpi2nc(mpireturn, "MPI_Allreduce");
            DEBUG_RETURN_ERROR(err)
        }
        if (status != NC_NOERR) DEBUG_RETURN_ERROR(status)
    }

    /* This function is called in ncmpi_enddef(), which can happen either when
     * creating a new file or opening an existing file with metadata modified.
     * For the former case, ncp->begin_var == 0 here.
     * For the latter case, we set begin_var a new value only if the new header
     * grows out of its extent or the start of non-record variables is not
     * aligned as requested by ncp->h_align.
     * Note ncp->xsz is header size and ncp->begin_var is header extent.
     * Add the minimum header free space requested by user.
     */

    //META: always assumes if vars are defined for now. Need MPI comm to tell whether all blocks has vars defined
    // if (ncp->vars.ndefined > 0)
    //     ncp->begin_var = D_RNDUP(ncp->xsz + ncp->h_minfree, ncp->h_align);
    // else /* no variable defined, ignore alignment and set header extent to
    //       * header size */
    //     ncp->begin_var = ncp->xsz;
    // printf("\nrank %d, ncp->xsz: %lld", rank, ncp->xsz);
    ncp->begin_var = D_RNDUP(ncp->xsz + ncp->h_minfree, ncp->h_align);
    //META:for debugging
    if (ncp->old != NULL) {
        /* If this define mode was entered from a redef(), we check whether
         * the new begin_var against the old begin_var. We do not shrink
         * the header extent.
         */
        if (ncp->begin_var < ncp->old->begin_var)
            ncp->begin_var = ncp->old->begin_var;
    }
    /* ncp->begin_var is the aligned starting file offset of the first
     * variable (also data section), which is the extent of file header
     * (header section). File extent may contain free space for header to grow.
     */

    /* Now calculate the starting file offsets for all variables.
     * loop thru vars, first pass is for the 'non-record' vars
     */
    NC_block *local_block;
    end_var = ncp->begin_var;

    for (int j=0; j<ncp->blocks.ndefined; j++){
        // if (rank == 1) printf("\n-1 rank %d, j: %lld", rank, j);
        local_block = ncp->blocks.value[j];
        if (local_block->modified){
            for (int i=0; i<ncp->blocks.value[j]->vars.ndefined; i++) {
                    /* skip record variables on this pass */
                    // if (rank == 1) printf("\nNC_begins: 1 rank %d, j: %lld", rank, j);
                    if (IS_RECVAR(local_block->vars.value[i])) continue;
                    if (first_var == NULL) first_var = local_block->vars.value[i];

                    /* for CDF-1 check if over the file size limit 32-bit integer */
                    if (ncp->format == 1 && end_var > NC_MAX_INT)
                        DEBUG_RETURN_ERROR(NC_EVARSIZE)
                    /* this will pad out non-record variables with the 4-byte alignment */
                    local_block->vars.value[i]->begin = D_RNDUP(end_var, 4);
                    // printf("\nNC_begins: local_block->vars.value[i]->begin: %lld", local_block->vars.value[i]->begin);
                    end_var = local_block->vars.value[i]->begin + local_block->vars.value[i]->len;
                }
        } else{
            end_var += local_block->block_var_len;
        }
    }



    /* only (re)calculate begin_rec if there is no sufficient space at end of
     * non-record variables or if the start of record variables is not aligned
     * as requested by ncp->r_align.
     */
    if (ncp->begin_rec < end_var + ncp->v_minfree)
        ncp->begin_rec = end_var + ncp->v_minfree;

    ncp->begin_rec = D_RNDUP(ncp->begin_rec, 4);

    /* align the starting offset for record variable section */
    if (ncp->r_align > 1)
        ncp->begin_rec = D_RNDUP(ncp->begin_rec, ncp->r_align);

    if (ncp->old != NULL) {
        /* check whether the new begin_rec is smaller */
        if (ncp->begin_rec < ncp->old->begin_rec)
            ncp->begin_rec = ncp->old->begin_rec;
    }

    if (first_var != NULL) ncp->begin_var = first_var->begin;
    else                   ncp->begin_var = ncp->begin_rec;

    end_var = ncp->begin_rec;
    /* end_var now is pointing to the beginning of record variables
     * note that this can be larger than the end of last non-record variable
     */
    //META: for record variables, also need to calibrate the record var begins based on block record var sizes
    for (int j=0; j<ncp->blocks.ndefined; j++){
        local_block = ncp->blocks.value[j];
        if (local_block->modified){
            for (int i=0; i<ncp->blocks.value[j]->vars.ndefined; i++){
                    /* skip non-record variables on this pass */
                    if (!IS_RECVAR(local_block->vars.value[i])) continue;
                    if (first_var == NULL) first_var = local_block->vars.value[i];

                    /* for CDF-1 check if over the file size limit 32-bit integer */
                    if (ncp->format == 1 && end_var > NC_MAX_INT)
                        DEBUG_RETURN_ERROR(NC_EVARSIZE)

                    local_block->vars.value[i]->begin = end_var;
                    // if (ncp->old != NULL) {
                    //     /* move to the next record variable */
                    //     for (; j<ncp->old->vars.ndefined; j++)
                    //         if (IS_RECVAR(ncp->old->vars.value[j]))
                    //             break;
                    //     if (j < ncp->old->vars.ndefined) {
                    //         if (ncp->vars.value[i]->begin < ncp->old->vars.value[j]->begin)
                    //             /* if the new begin is smaller, use the old begin */
                    //             ncp->vars.value[i]->begin = ncp->old->vars.value[j]->begin;
                    //         j++;
                    //     }
                    // }
                    end_var += local_block->vars.value[i]->len;
                }
        } else{
            end_var += local_block->block_recvar_len;
        }
    }


/* below is only needed if alignment is performed on record variables */
#if 0
    /*
     * for special case of exactly one record variable, pack value
     */
    /* if there is exactly one record variable, then there is no need to
     * pad for alignment -- there's nothing after it */
    if (last != NULL && ncp->recsize == last->len)
        ncp->recsize = *last->dsizes * last->xsz;
#endif

    if (NC_IsNew(ncp)) ncp->numrecs = 0;
    return NC_NOERR;
}


void print_buffer_ascii(const unsigned char* buffer, size_t length) {
    for (size_t i = 0; i < length; i++) {
        if (isprint(buffer[i])) {
            printf("%c", buffer[i]);
        } else {
            printf(".");
        }
    }
    printf("\n");
}

/*----< write_NC() >---------------------------------------------------------*/
/*
 * This function is collective and only called by enddef().
 * Write out the header
 * 1. Call ncmpio_hdr_put_NC() to copy the header object, ncp, to a buffer.
 * 2. Process rank 0 writes the header to file.
 */
static int
write_NC(NC *ncp)
{
    int status=NC_NOERR, mpireturn, err, rank;
    MPI_Offset i, global_header_wlen, local_header_wlen, ntimes;
    MPI_Status mpistatus;

    assert(!NC_readonly(ncp));

    MPI_Comm_rank(ncp->comm, &rank);

    /* In NC_begins(), root's ncp->xsz and ncp->begin_var, root's header
     * size and extent, have been broadcast (sync-ed) among processes.
     */
//TODO: decide if need to do this for new file format
// #ifdef ENABLE_NULL_BYTE_HEADER_PADDING
//     /* NetCDF classic file formats require the file header null-byte padded.
//      * PnetCDF's default is not to write the padding area (between ncp->xsz and
//      * ncp->begin_var). When this padding feature is enabled, we write the
//      * padding area only when writing the header the first time, i.e. creating
//      * a new file, or the new header extent becomes larger than the old one.
//      */
//     if (ncp->old == NULL || ncp->begin_var > ncp->old->begin_var)
//         header_wlen = ncp->begin_var;
//     else
//         header_wlen = ncp->xsz;
// #else
    // /* Do not write padding area (between ncp->xsz and ncp->begin_var) */
    // header_wlen = ncp->xsz;
// #endif
    global_header_wlen = ncp->global_xsz;
    global_header_wlen = _RNDUP(global_header_wlen, X_ALIGN);
    

    /* if header_wlen is > NC_MAX_INT, then write the header in chunks.
     * Note reading file header is already done in chunks. See
     * ncmpio_hdr_get_NC().
     */
    ntimes = global_header_wlen / NC_MAX_INT;
    if (global_header_wlen % NC_MAX_INT) ntimes++;

    /* only rank 0's header gets written to the file */
    /*META new header file: only rank 0 writes global header */
    MPI_Offset offset, remain;
    if (rank == 0) {
        char *buf=NULL, *buf_ptr;

#ifdef ENABLE_NULL_BYTE_HEADER_PADDING
        /* NetCDF classic file formats require the file header null-byte
         * padded. Thus we must calloc a buffer of size equal to file header
         * extent.
         */
        buf = (char*)NCI_Calloc(global_header_wlen, 1);
#else
        // printf("\nglobal_header_wlen: %lld", global_header_wlen);
        /* Do not write padding area (between ncp->xsz and ncp->begin_var) */
        buf = (char*)NCI_Malloc(global_header_wlen);
#endif
        /* copy the entire global header object to buf */
        status = ncmpio_global_hdr_put_NC(ncp, buf);
        // print_buffer_ascii(buf, global_header_wlen);

        if (status != NC_NOERR) /* a fatal error */
            goto fn_exit;

        /* For non-fatal error, we continue to write header to the file, as now
         * the header object in memory has been sync-ed across all processes.
         */

        /* rank 0's fileview already includes the file header */

        /* explicitly initialize mpistatus object to 0. For zero-length read,
         * MPI_Get_count may report incorrect result for some MPICH version,
         * due to the uninitialized MPI_Status object passed to MPI-IO calls.
         * Thus we initialize it above to work around.
         */
        memset(&mpistatus, 0, sizeof(MPI_Status));

        /* write the header in chunks */
        offset = 0;
        remain = global_header_wlen;
        buf_ptr = buf;
        for (i=0; i<ntimes; i++) {
            int bufCount = (int) MIN(remain, NC_MAX_INT);
            // printf("\nwrite global header at offset %lld, bufCount: %d", offset, bufCount);
            if (fIsSet(ncp->flags, NC_HCOLL))
                TRACE_IO(MPI_File_write_at_all)(ncp->collective_fh, offset, buf_ptr,
                                                bufCount, MPI_BYTE, &mpistatus);
            else
                TRACE_IO(MPI_File_write_at)(ncp->collective_fh, offset, buf_ptr,
                                            bufCount, MPI_BYTE, &mpistatus);
            if (mpireturn != MPI_SUCCESS) {
                err = ncmpii_error_mpi2nc(mpireturn, "MPI_File_write_at");
                /* write has failed, which is more serious than inconsistency */
                if (err == NC_EFILE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
            }
            else {
                /* Update the number of bytes read since file open.
                 * Because each rank writes no more than NC_MAX_INT at a time,
                 * it is OK to call MPI_Get_count, instead of MPI_Get_count_c.
                 */
                int put_size;
                mpireturn = MPI_Get_count(&mpistatus, MPI_BYTE, &put_size);
                if (mpireturn != MPI_SUCCESS || put_size == MPI_UNDEFINED)
                    ncp->put_size += bufCount;
                else
                    ncp->put_size += put_size;
            }
            offset  += bufCount;
            buf_ptr += bufCount;
            remain  -= bufCount;
        }
        NCI_Free(buf);

    }
    else if (fIsSet(ncp->flags, NC_HCOLL)) {
        /* other processes participate the collective call */

        for (i=0; i<ntimes; i++)
            TRACE_IO(MPI_File_write_at_all)(ncp->collective_fh, 0, NULL,
                                            0, MPI_BYTE, &mpistatus);
    }

    //META: all processes write the local header

    //Create viewtype and datatype to write (multiple) blocks with one MPI_I/O call
    int num_blocks_w = 0;
    for (int i = 0; i < ncp->blocks.ndefined; i++){

        if (ncp->blocks.value[i]->modified)
            num_blocks_w++;
    }

    if (num_blocks_w > 0){
        MPI_Datatype memtype;
        MPI_Aint *memdisps = (MPI_Aint*)NCI_Malloc(num_blocks_w * sizeof(MPI_Aint));
        char **local_bufs = (char**)NCI_Malloc(num_blocks_w * sizeof(char*));
        MPI_Datatype filetype;
        MPI_Aint *filedisps = (MPI_Aint*)NCI_Malloc(num_blocks_w * sizeof(MPI_Aint));
        int *blocklens = (int*)NCI_Malloc(num_blocks_w * sizeof(int));
        int j = 0;
        memdisps[0] = 0;
        for (int i = 0; i < ncp->blocks.ndefined; i++){
            if (ncp->blocks.value[i]->modified){
                local_header_wlen = _RNDUP(ncp->blocks.value[i]->xsz, X_ALIGN);
                blocklens[j] = local_header_wlen;
                local_bufs[j] = (char*)NCI_Malloc(local_header_wlen);
                status = ncmpio_local_hdr_put_NC(ncp, local_bufs[j], i);
                if (status != NC_NOERR) /* a fatal error */
                    goto fn_exit;
                filedisps[j] = ncp->blocks.value[i]->begin;
                if (j > 0) memdisps[j] = local_bufs[j] - local_bufs[0];
                j++;
            }
        }
        // for (int i = 0; i < num_blocks_w; i++){
        //     printf("\nrank %d, blocklens[%d]: %d, filedisps[%d]: %lld", rank, i, blocklens[i], i, filedisps[i]);
        // }   
        MPI_Type_create_hindexed(num_blocks_w, blocklens, memdisps, MPI_BYTE, &memtype);
        MPI_Type_commit(&memtype);
        NCI_Free(memdisps);
        

        MPI_Type_create_hindexed(num_blocks_w, blocklens, filedisps, MPI_BYTE, &filetype);
        MPI_Type_commit(&filetype);
        NCI_Free(blocklens);
        NCI_Free(filedisps);


        TRACE_IO(MPI_File_set_view)(ncp->collective_fh, 0, MPI_BYTE, filetype, "native", MPI_INFO_NULL);
        MPI_Type_free(&filetype);
        if (fIsSet(ncp->flags, NC_HCOLL)) 
            TRACE_IO(MPI_File_write_at_all_c)(ncp->collective_fh, 0, local_bufs[0], 1, memtype, &mpistatus);
        else
            TRACE_IO(MPI_File_write_at_c)(ncp->collective_fh, 0, local_bufs[0], 1, memtype, &mpistatus);
        MPI_Type_free(&memtype);
        for (int i = 0; i < num_blocks_w; i++) NCI_Free(local_bufs[i]);
        NCI_Free(local_bufs);
    } else {
        /* other processes participate the collective call */
        MPI_Datatype emptytype;
        MPI_Type_contiguous(0, MPI_BYTE, &emptytype);
        MPI_Type_commit(&emptytype);
        TRACE_IO(MPI_File_set_view)(ncp->collective_fh, 0, MPI_BYTE, emptytype, "native", MPI_INFO_NULL);
        MPI_Type_free(&emptytype);
        if (fIsSet(ncp->flags, NC_HCOLL))
            TRACE_IO(MPI_File_write_at_all_c)(ncp->collective_fh, 0, NULL, 0, MPI_BYTE, &mpistatus);
    }
        
        
    





    /* copy the  local header object to buf */
    // for (int i = 0; i < ncp->blocks.ndefined; i++) {
    //     if (!ncp->blocks.value[i]->modified){
    //         //TODO: unmodified blocks can still be written to file if they are moved
    //         continue;
    //     } //skip the block that hasn't been modified
    //     char *local_buf=NULL, *local_buf_ptr;
    //     local_header_wlen = _RNDUP(ncp->blocks.value[i]->xsz, X_ALIGN);
    //     local_buf = (char*)NCI_Malloc(local_header_wlen);
    //     status = ncmpio_local_hdr_put_NC(ncp, local_buf, i);
    //     if (status != NC_NOERR) /* a fatal error */
    //         goto fn_exit;

    //     /* For non-fatal error, we continue to write header to the file, as now
    //     * the header object in memory has been sync-ed across all processes.
    //     */

    //     /* rank 0's fileview already includes the file header */

    //     /* explicitly initialize mpistatus object to 0. For zero-length read,
    //     * MPI_Get_count may report incorrect result for some MPICH version,
    //     * due to the uninitialized MPI_Status object passed to MPI-IO calls.
    //     * Thus we initialize it above to work around.
    //     */
    //     memset(&mpistatus, 0, sizeof(MPI_Status));

    //     /* write the header in chunks */
    //     offset = ncp->blocks.value[i]->begin;
    //     remain = local_header_wlen;
    //     local_buf_ptr = local_buf;
    //     ntimes = local_header_wlen / NC_MAX_INT;
    //     if (local_header_wlen % NC_MAX_INT) ntimes++;  
    //     for (j=0; j<ntimes; j++) {
    //         int bufCount = (int) MIN(remain, NC_MAX_INT);
    //         if (fIsSet(ncp->flags, NC_HCOLL))
    //             TRACE_IO(MPI_File_write_at_all)(ncp->collective_fh, offset, local_buf_ptr,
    //                                             bufCount, MPI_BYTE, &mpistatus);
    //         else
    //             TRACE_IO(MPI_File_write_at)(ncp->collective_fh, offset, local_buf_ptr,
    //                                         bufCount, MPI_BYTE, &mpistatus);
    //         if (mpireturn != MPI_SUCCESS) {
    //             err = ncmpii_error_mpi2nc(mpireturn, "MPI_File_write_at");
    //             /* write has failed, which is more serious than inconsistency */
    //             if (err == NC_EFILE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
    //         }
    //         else {
    //             /* Update the number of bytes read since file open.
    //             * Because each rank writes no more than NC_MAX_INT at a time,
    //             * it is OK to call MPI_Get_count, instead of MPI_Get_count_c.
    //             */
    //             int put_size;
    //             mpireturn = MPI_Get_count(&mpistatus, MPI_BYTE, &put_size);
    //             if (mpireturn != MPI_SUCCESS || put_size == MPI_UNDEFINED)
    //                 ncp->put_size += bufCount;
    //             else
    //                 ncp->put_size += put_size;
    //         }
    //         offset  += bufCount;
    //         local_buf_ptr += bufCount;
    //         remain  -= bufCount;
    //     }
    //     NCI_Free(local_buf);
    // }
        

fn_exit:
    if (ncp->safe_mode == 1) {
        /* broadcast root's status, because only root writes to the file */
        int root_status = status;
        TRACE_COMM(MPI_Bcast)(&root_status, 1, MPI_INT, 0, ncp->comm);
        /* root's write has failed, which is more serious than inconsistency */
        if (root_status == NC_EWRITE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
    }

    fClr(ncp->flags, NC_NDIRTY);

    return status;
}

/* Many subroutines called in ncmpio__enddef() are collective. We check the
 * error codes of all processes only in safe mode, so the program can stop
 * collectively, if any one process got an error. However, when safe mode is
 * off, we simply return the error and program may hang if some processes
 * do not get error and proceed to the next subroutine call.
 */
#define CHECK_ERROR(err) {                                              \
    if (ncp->safe_mode == 1) {                                          \
        int status;                                                     \
        TRACE_COMM(MPI_Allreduce)(&err, &status, 1, MPI_INT, MPI_MIN,   \
                                  ncp->comm);                           \
        if (mpireturn != MPI_SUCCESS) {                                 \
            err = ncmpii_error_mpi2nc(mpireturn, "MPI_Allreduce");      \
            DEBUG_RETURN_ERROR(err)                                     \
        }                                                               \
        if (status != NC_NOERR) return status;                          \
    }                                                                   \
    else if (err != NC_NOERR)                                           \
        return err;                                                     \
}

/*----< ncmpio_NC_check_vlen() >---------------------------------------------*/
/* Check whether variable size is less than or equal to vlen_max,
 * without overflowing in arithmetic calculations.  If OK, return 1,
 * else, return 0.  For CDF1 format or for CDF2 format on non-LFS
 * platforms, vlen_max should be 2^31 - 4, but for CDF2 format on
 * systems with LFS it should be 2^32 - 4.
 */
int
ncmpio_NC_check_vlen(NC_var     *varp,
                     MPI_Offset  vlen_max)
{
    int i;
    MPI_Offset prod=varp->xsz;     /* product of xsz and dimensions so far */

    for (i = IS_RECVAR(varp) ? 1 : 0; i < varp->ndims; i++) {
        if (varp->shape[i] > vlen_max / prod) {
            return 0;           /* size in bytes > vlen_max */
        }
        prod *= varp->shape[i];
    }
    return 1;
}

/*----< ncmpio_NC_check_vlens() >--------------------------------------------*/
/* Given a valid ncp, check all variables for their sizes against the maximal
 * allowable sizes. Different CDF formation versions have different maximal
 * sizes. This function returns NC_EVARSIZE if any variable has a bad len
 * (product of non-rec dim sizes too large), else return NC_NOERR.
 */
int
ncmpio_NC_check_vlens(NC *ncp)
{
    int last = 0;
    MPI_Offset i, vlen_max, rec_vars_count;
    MPI_Offset large_fix_vars_count, large_rec_vars_count;
    NC_var *varp;

    if (ncp->vars.ndefined == 0) /* no variable defined */
        return NC_NOERR;

    /* maximum permitted variable size (or size of one record's worth
       of a record variable) in bytes. It is different between format 1
       2 and 5. */

    if (ncp->format >= 5) /* CDF-5 format max */
        vlen_max = NC_MAX_INT64 - 3; /* "- 3" handles rounded-up size */
    else if (ncp->format == 2) /* CDF2 format */
        vlen_max = NC_MAX_UINT  - 3; /* "- 3" handles rounded-up size */
    else
        vlen_max = NC_MAX_INT   - 3; /* CDF1 format */

    /* Loop through vars, first pass is for non-record variables */
    large_fix_vars_count = 0;
    rec_vars_count = 0;
    for (i=0; i<ncp->vars.ndefined; i++) {
        varp = ncp->vars.value[i];
        if (IS_RECVAR(varp)) {
            rec_vars_count++;
            continue;
        }

        last = 0;
        if (ncmpio_NC_check_vlen(varp, vlen_max) == 0) {
            /* check this variable's shape product against vlen_max */

            if (ncp->format >= 5) /* variable too big for CDF-5 */
                DEBUG_RETURN_ERROR(NC_EVARSIZE)

            large_fix_vars_count++;
            last = 1;
        }
    }
    /* OK if last non-record variable size too large, since not used to
       compute an offset */
    if (large_fix_vars_count > 1)  /* only one "too-large" variable allowed */
        DEBUG_RETURN_ERROR(NC_EVARSIZE)

    /* The only "too-large" variable must be the last one defined */
    if (large_fix_vars_count == 1 && last == 0)
        DEBUG_RETURN_ERROR(NC_EVARSIZE)

    if (rec_vars_count == 0) return NC_NOERR;

    /* if there is a "too-large" fixed-size variable, no record variable is
     * allowed */
    if (large_fix_vars_count == 1)
        DEBUG_RETURN_ERROR(NC_EVARSIZE)

    /* Loop through vars, second pass is for record variables.   */
    large_rec_vars_count = 0;
    for (i=0; i<ncp->vars.ndefined; i++) {
        varp = ncp->vars.value[i];
        if (!IS_RECVAR(varp)) continue;

        last = 0;
        if (ncmpio_NC_check_vlen(varp, vlen_max) == 0) {
            /* check this variable's shape product against vlen_max */

            if (ncp->format >= 5) /* variable too big for CDF-5 */
                DEBUG_RETURN_ERROR(NC_EVARSIZE)

            large_rec_vars_count++;
            last = 1;
        }
    }

    /* For CDF-2, no record variable can require more than 2^32 - 4 bytes of
     * storage for each record's worth of data, unless it is the last record
     * variable. See
     * http://www.unidata.ucar.edu/software/netcdf/docs/file_structure_and_performance.html#offset_format_limitations
     */
    if (large_rec_vars_count > 1)  /* only one "too-large" variable allowed */
        DEBUG_RETURN_ERROR(NC_EVARSIZE)

    /* and it has to be the last one */
    if (large_rec_vars_count == 1 && last == 0)
        DEBUG_RETURN_ERROR(NC_EVARSIZE)

    return NC_NOERR;
}

#ifdef VAR_BEGIN_IN_ARBITRARY_ORDER
typedef struct {
    MPI_Offset off;      /* starting file offset of a variable */
    MPI_Offset len;      /* length in bytes of a variable */
    int        ID;       /* variable index ID */
} off_len;

/*----< off_compare() >------------------------------------------------------*/
/* used for sorting the offsets of the off_len array */
static int
off_compare(const void *a, const void *b)
{
    if (((off_len*)a)->off > ((off_len*)b)->off) return  1;
    if (((off_len*)a)->off < ((off_len*)b)->off) return -1;
    return 0;
}
#endif

/*----< ncmpio_NC_check_voffs() >--------------------------------------------*/
/*
 * Given a valid ncp, check whether the file starting offsets (begin) of all
 * variables follows the same increasing order as they were defined.
 *
 * In NetCDF User's Guide, Chapter "File Structure and Performance", Section
 * "Parts of a NetCDF Class File", the following statement implies such
 * checking. "The order in which the variable data appears in each data section
 * is the same as the order in which the variables were defined, in increasing
 * numerical order by netCDF variable ID." URLs are given below.
 * https://www.unidata.ucar.edu/software/netcdf/documentation/historic/netcdf/Classic-File-Parts.html
 * https://www.unidata.ucar.edu/software/netcdf/docs/file_structure_and_performance.html#classic_file_parts
 *
 * However, the CDF file format specification does not require such order.
 * NetCDF version 4.6.0 and priors do not enforce this checking, but all
 * assume this requirement. See subroutine NC_computeshapes() in libsrc/v1hpg.c.
 * Similarly for PnetCDF, this check was not enforced until 1.9.0. Therefore,
 * it is important to keep this check to avoid potential problems.
 *
 * It appears that python scipy.netcdf does not follow this. An example can be
 * found in the NetCDF discussion thread:
 * https://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2018/msg00050.html
 * A scipy.netcdf program opens a NetCDF file with a few variables already
 * defined, enters define mode through redef call, and adds a new variable. The
 * scipy.netcdf implementation probably chooses to insert the new variable
 * entry in the front of "var_list" in file header. Technically speaking, this
 * does not violate the classic file format specification, but may result in
 * the file starting offsets ("begin" entry) of all variables defined in the
 * file header failed to appear in an increasing order. To obtain the file
 * header extent, one must scan the "begin" entry of all variables and find the
 * minimum as the extent.
 */
int
ncmpio_NC_check_voffs(NC *ncp)
{
    int i, num_fix_vars, prev;
    MPI_Offset prev_off;

    if (ncp->vars.ndefined == 0) return NC_NOERR;

    num_fix_vars = ncp->vars.ndefined - ncp->vars.num_rec_vars;

#ifdef VAR_BEGIN_IN_ARBITRARY_ORDER
    int j;
    off_len *var_off_len;
    MPI_Offset var_end, max_var_end;

    if (num_fix_vars == 0) goto check_rec_var;

    /* check non-record variables first */
    var_off_len = (off_len*) NCI_Malloc(num_fix_vars * sizeof(off_len));
    for (i=0, j=0; i<ncp->vars.ndefined; i++) {
        NC_var *varp = ncp->vars.value[i];
        if (varp->begin < ncp->xsz) {
            if (ncp->safe_mode) {
                printf("Variable %s begin offset (%lld) is less than file header extent (%lld)\n",
                       varp->name, varp->begin, ncp->xsz);
            }
            NCI_Free(var_off_len);
            DEBUG_RETURN_ERROR(NC_ENOTNC)
        }
        if (IS_RECVAR(varp)) continue;
        var_off_len[j].off = varp->begin;
        var_off_len[j].len = varp->len;
        var_off_len[j].ID  = i;
        j++;
    }
    assert(j == num_fix_vars);

    for (i=1; i<num_fix_vars; i++) {
        if (var_off_len[i].off < var_off_len[i-1].off)
            break;
    }

    if (i < num_fix_vars)
        /* sort the off-len array into an increasing order */
        qsort(var_off_len, num_fix_vars, sizeof(off_len), off_compare);

    max_var_end = var_off_len[0].off + var_off_len[0].len;
    for (i=1; i<num_fix_vars; i++) {
        if (var_off_len[i].off < var_off_len[i-1].off + var_off_len[i-1].len) {
            if (ncp->safe_mode) {
                NC_var *var_cur = ncp->vars.value[var_off_len[i].ID];
                NC_var *var_prv = ncp->vars.value[var_off_len[i-1].ID];
                printf("Variable %s begin offset (%lld) overlaps variable %s (begin=%lld, length=%lld)\n",
                       var_cur->name, var_cur->begin, var_prv->name, var_prv->begin, var_prv->len);
            }
            NCI_Free(var_off_len);
            DEBUG_RETURN_ERROR(NC_ENOTNC)
        }
        var_end = var_off_len[i].off + var_off_len[i].len;
        max_var_end = MAX(max_var_end, var_end);
    }

    if (ncp->begin_rec < max_var_end) {
        if (ncp->safe_mode)
            printf("Record variable section begin (%lld) is less than fixed-size variable section end (%lld)\n",
                   ncp->begin_rec, max_var_end);
        NCI_Free(var_off_len);
        DEBUG_RETURN_ERROR(NC_ENOTNC)
    }
    NCI_Free(var_off_len);

check_rec_var:
    if (ncp->vars.num_rec_vars == 0) return NC_NOERR;

    /* check record variables */
    var_off_len = (off_len*) NCI_Malloc(ncp->vars.num_rec_vars * sizeof(off_len));
    for (i=0, j=0; i<ncp->vars.ndefined; i++) {
        NC_var *varp = ncp->vars.value[i];
        if (!IS_RECVAR(varp)) continue;
        var_off_len[j].off = varp->begin;
        var_off_len[j].len = varp->len;
        var_off_len[j].ID  = i;
        j++;
    }
    assert(j == ncp->vars.num_rec_vars);

    for (i=1; i<ncp->vars.num_rec_vars; i++) {
        if (var_off_len[i].off < var_off_len[i-1].off)
            break;
    }

    if (i < ncp->vars.num_rec_vars)
        /* sort the off-len array into an increasing order */
        qsort(var_off_len, ncp->vars.num_rec_vars, sizeof(off_len), off_compare);

    for (i=1; i<ncp->vars.num_rec_vars; i++) {
        if (var_off_len[i].off < var_off_len[i-1].off + var_off_len[i-1].len) {
            if (ncp->safe_mode) {
                NC_var *var_cur = ncp->vars.value[var_off_len[i].ID];
                NC_var *var_prv = ncp->vars.value[var_off_len[i-1].ID];
                printf("Variable %s begin offset (%lld) overlaps variable %s (begin=%lld, length=%lld)\n",
                       var_cur->name, var_cur->begin, var_prv->name, var_prv->begin, var_prv->len);
            }
            NCI_Free(var_off_len);
            DEBUG_RETURN_ERROR(NC_ENOTNC)
        }
    }
    NCI_Free(var_off_len);
#else
    /* Loop through vars, first pass is for non-record variables */
    if (num_fix_vars == 0) goto check_rec_var;

    prev = 0;
    prev_off = ncp->begin_var;

    for (i=0; i<ncp->vars.ndefined; i++) {
        NC_var *varp = ncp->vars.value[i];
        if (IS_RECVAR(varp)) continue;

        if (varp->begin < prev_off) {
            if (ncp->safe_mode) {
                if (i == 0)
                    printf("Variable \"%s\" begin offset (%lld) is less than header extent (%lld)\n",
                           varp->name, varp->begin, prev_off);
                else
                    printf("Variable \"%s\" begin offset (%lld) is less than previous variable \"%s\" end offset (%lld)\n",
                           varp->name, varp->begin, ncp->vars.value[prev]->name, prev_off);
            }
            DEBUG_RETURN_ERROR(NC_ENOTNC)
        }
        prev_off = varp->begin + varp->len;
        prev     = i;
    }

    if (ncp->begin_rec < prev_off) {
        if (ncp->safe_mode)
            printf("Record variable section begin offset (%lld) is less than fixed-size variable section end offset (%lld)\n",
                   ncp->begin_rec, prev_off);
        DEBUG_RETURN_ERROR(NC_ENOTNC)
    }

check_rec_var:
    if (ncp->vars.num_rec_vars == 0) return NC_NOERR;

    /* Loop through vars, second pass is for record variables */
    prev_off = ncp->begin_rec;
    prev     = 0;
    for (i=0; i<ncp->vars.ndefined; i++) {
        NC_var *varp = ncp->vars.value[i];
        if (!IS_RECVAR(varp)) continue;

        if (varp->begin < prev_off) {
            if (ncp->safe_mode) {
                printf("Variable \"%s\" begin offset (%lld) is less than previous variable end offset (%lld)\n",
                           varp->name, varp->begin, prev_off);
                if (i == 0)
                    printf("Variable \"%s\" begin offset (%lld) is less than record variable section begin offset (%lld)\n",
                           varp->name, varp->begin, prev_off);
                else
                    printf("Variable \"%s\" begin offset (%lld) is less than previous variable \"%s\" end offset (%lld)\n",
                           varp->name, varp->begin, ncp->vars.value[prev]->name, prev_off);
            }
            DEBUG_RETURN_ERROR(NC_ENOTNC)
        }
        prev_off = varp->begin + varp->len;
        prev     = i;
    }
#endif

    return NC_NOERR;
}

static int
NC_begins_local(NC *ncp)
{   int i;
    NC_var *last = NULL;
    for (i=0; i<ncp->blocks.ndefined; i++) {
        ncp->blocks.value[i]->xsz = ncmpio_block_hdr_len_NC(ncp, i);
    }

    //META: calculate block var size: non-record var sizes in data section
    MPI_Offset begin_tmp = 0;
    for (i=0; i<ncp->blocks.ndefined; i++) {
        ncp->blocks.value[i]->block_var_len = 0;
        for(int j=0; j<ncp->blocks.value[i]->vars.ndefined; j++){
            if (IS_RECVAR(ncp->blocks.value[i]->vars.value[j])) continue;
            begin_tmp =  D_RNDUP(begin_tmp + ncp->blocks.value[i]->vars.value[j]->len, 4);
        }
        ncp->blocks.value[i]->block_var_len = begin_tmp;
    }

    //META: calculate block var size: record var sizes in data section
    for (i=0; i<ncp->blocks.ndefined; i++) {
        ncp->blocks.value[i]->block_recvar_len = 0;
        for(int j=0; j<ncp->blocks.value[i]->vars.ndefined; j++){
            if (!IS_RECVAR(ncp->blocks.value[i]->vars.value[j])) continue;
#if SIZEOF_OFF_T == SIZEOF_SIZE_T && SIZEOF_SIZE_T == 4
        if (cp->blocks.value[i]->block_recvar_len > NC_MAX_UINT - ncp->blocks.value[i]->vars.value[j]->len)
            DEBUG_RETURN_ERROR(NC_EVARSIZE)
#endif
            ncp->blocks.value[i]->block_recvar_len += ncp->blocks.value[i]->vars.value[j]->len;
            last = ncp->blocks.value[i]->vars.value[j];
        }
    }
    /*
     * for special case (Check CDF-1 and CDF-2 file format specifications.)
     * "A special case: Where there is exactly one record variable, we drop the
     * requirement that each record be four-byte aligned, so in this case there
     * is no record padding."
     */
    if (last != NULL) {
        if (ncp->blocks.value[i]->block_recvar_len == last->len) {
            /* exactly one record variable, pack value */
            ncp->blocks.value[i]->block_recvar_len = *last->dsizes * last->xsz;
        }
#if 0
        else if (last->len == UINT32_MAX) { /* huge last record variable */
            ncp->recsize += *last->dsizes * last->xsz;
        }
#endif
    }
    return NC_NOERR;
}

/*----< ncmpio__ >---------------------------------------------------*/
/* This is a collective subroutine.
 * h_minfree  Sets the pad at the end of the "header" section, i.e. at least
 *            this amount of free space includes at the end of header extent.
 * v_align    Controls the alignment of the beginning of the data section for
 *            fixed size variables.
 * v_minfree  Sets the pad at the end of the data section for fixed size
 *            variables, i.e. at least this amount of free space between the
 *            fixed-size variable section and record variable section.
 * r_align    Controls the alignment of the beginning of the data section for
 *            variables which have an unlimited dimension (record variables).
 */
int
ncmpio__enddef(void       *ncdp,
               MPI_Offset  h_minfree,
               MPI_Offset  v_align,
               MPI_Offset  v_minfree,
               MPI_Offset  r_align)
{
    int i, num_fix_vars, mpireturn, err=NC_NOERR, status=NC_NOERR;
    int rank, nproc;
    char value[MPI_MAX_INFO_VAL];
    NC *ncp = (NC*)ncdp;
    
    // printf("\npncp->ncp->xsz: %lld\n", ncp->xsz);
    

    /* negative values of h_minfree, v_align, v_minfree, r_align have been
     * checked at dispatchers.
     */

    /* sanity check for NC_ENOTINDEFINE, NC_EINVAL, NC_EMULTIDEFINE_FNC_ARGS
     * has been done at dispatchers */
    ncp->h_minfree = h_minfree;
    ncp->v_minfree = v_minfree;

    /* calculate a good file extent alignment size based on:
     *     + hints set by users in the environment variable PNETCDF_HINTS
     *       nc_header_align_size and nc_var_align_size
     *     + v_align set in the call to ncmpi__enddef()
     * Hints set in the environment variable PNETCDF_HINTS have the higher
     * precedence than the ones set in the API calls.
     * The precedence of hints and arguments:
     * 1. hints set in PNETCDF_HINTS environment variable at run time
     * 2. hints set in the source codes, for example, a call to
     *    MPI_Info_set("nc_header_align_size", "1048576");
     * 3. source codes calling ncmpi__enddef(). For example,
     *    MPI_Offset v_align = 1048576;
     *    ncmpi__enddef(ncid, 0, v_align, 0, 0);
     * 4. defaults
     *       0   for h_minfree
     *       512 for v_align
     *       0   for v_minfree
     *       4   for r_align
     */

    /* ncp->h_align, ncp->v_align, ncp->r_align, and ncp->chunk have been
     * set during file create/open */

    num_fix_vars = ncp->vars.ndefined - ncp->vars.num_rec_vars;

    /* reset to hints set at file create/open time */
    ncp->h_align = ncp->env_h_align;
    ncp->v_align = ncp->env_v_align;
    ncp->r_align = ncp->env_r_align;

    if (ncp->h_align == 0) {   /* hint nc_header_align_size is not set */
        if (ncp->v_align > 0)  /* hint nc_var_align_size is set */
            ncp->h_align = ncp->v_align;
        else if (v_align > 0)  /* v_align is passed from ncmpi__enddef */
            ncp->h_align = v_align;

        /* if no fixed-size variables is defined, use r_align */
        if (ncp->h_align == 0 && num_fix_vars == 0) {
            if (ncp->r_align > 0)  /* hint nc_record_align_size is set */
                ncp->h_align = ncp->r_align;
            else if (r_align > 0)  /* r_align is passed from ncmpi__enddef */
                ncp->h_align = r_align;
        }

        if (ncp->h_align == 0 && ncp->old == NULL)
            /* h_align is still not set. Set h_align only when creating a new
             * file. When opening an existing file file, setting h_align here
             * may unexpectedly grow the file extent.
             */
            ncp->h_align = FILE_ALIGNMENT_DEFAULT;
    }
    /* else respect user hint */

    if (ncp->v_align == 0) { /* user info does not set nc_var_align_size */
        if (v_align > 0)     /* v_align is passed from ncmpi__enddef */
            ncp->v_align = v_align;
        /* else ncp->v_align is already set by user/env, ignore the one passed
         * by the argument v_align of this subroutine.
         */
    }

    if (ncp->r_align == 0) { /* user info does not set nc_record_align_size */
        if (r_align > 0)     /* r_align is passed from ncmpi__enddef */
            ncp->r_align = r_align;
        /* else ncp->r_align is already set by user/env, ignore the one passed
         * by the argument r_align of this subroutine.
         */
    }

    /* all CDF formats require 4-bytes alignment */
    if (ncp->h_align == 0)    ncp->h_align = 4;
    else                      ncp->h_align = D_RNDUP(ncp->h_align, 4);
    if (ncp->v_align == 0)    ncp->v_align = 4;
    else                      ncp->v_align = D_RNDUP(ncp->v_align, 4);
    if (ncp->r_align == 0)    ncp->r_align = 4;
    else                      ncp->r_align = D_RNDUP(ncp->r_align, 4);

    /* reflect the hint changes to the MPI info object, so the user can inquire
     * what the true hint values are being used
     */
    sprintf(value, "%lld", ncp->h_align);
    MPI_Info_set(ncp->mpiinfo, "nc_header_align_size", value);
    sprintf(value, "%lld", ncp->v_align);
    MPI_Info_set(ncp->mpiinfo, "nc_var_align_size", value);
    sprintf(value, "%lld", ncp->r_align);
    MPI_Info_set(ncp->mpiinfo, "nc_record_align_size", value);

#ifdef ENABLE_SUBFILING
    sprintf(value, "%d", ncp->num_subfiles);
    MPI_Info_set(ncp->mpiinfo, "nc_num_subfiles", value);
    if (ncp->num_subfiles > 1) {
        /* TODO: should return subfile-related msg when there's an error */
        err = ncmpio_subfile_partition(ncp);
        CHECK_ERROR(err)
    }
#else
    MPI_Info_set(ncp->mpiinfo, "pnetcdf_subfiling", "disable");
    MPI_Info_set(ncp->mpiinfo, "nc_num_subfiles", "0");
#endif

    /* check whether sizes of all variables are legal */
    err = ncmpio_NC_check_vlens(ncp);
    CHECK_ERROR(err)

    /* When ncp->old == NULL, this enddef is called the first time after file
     * create call. In this case, we compute each variable's 'begin', starting
     * file offset as well as the offsets of record variables.
     * When ncp->old != NULL, this enddef is called after a redef. In this
     * case, we re-used all variable offsets as many as possible.
     *
     * Note in NC_begins, root broadcasts ncp->xsz, the file header size, to
     * all processes.
     */

    
    //META: calculate block size
    NC_begins_local(ncp);
    CHECK_ERROR(err)



    //META:Merge block arrays: collect all modified blocks from all processes
    
    MPI_Comm_rank(ncp->comm, &rank);
    MPI_Comm_size(ncp->comm, &nproc);

    // if (rank == 1) printf("\nbefore comm rank %d: ncp->blocks.value[0]->vars.ndefined: %d\n", rank, ncp->blocks.value[0]->vars.ndefined);


    //STEP1: communicate number of new blocks across all processes
    int num_new_blocks;
    num_new_blocks = ncp->blocks.ndefined - ncp->blocks.nread;
    int* all_num_news = (int*) NCI_Malloc(nproc * X_SIZEOF_INT);

    // Collect local header sizes from all processes

    TRACE_COMM(MPI_Allgather)(&num_new_blocks, 1, MPI_INT, all_num_news, 1, MPI_INT, ncp->comm);
    int total_new_blocks = 0;
    int* new_block_offsets = (int*) NCI_Malloc(nproc * X_SIZEOF_INT);
    new_block_offsets[0] = 0;
    for (int i = 0; i < nproc; i++) {
        total_new_blocks += all_num_news[i];
        if(i>0) new_block_offsets[i] = new_block_offsets[i-1] + all_num_news[i-1];
        }

    //STEP2: communicate all modified blocks content across all processes
    MPI_Offset local_buff_size;
    local_buff_size = hdr_len_NC_modified_blockarray(&ncp->blocks);
    char* local_buff = (char*) NCI_Malloc(local_buff_size);
    err = serialize_bufferinfo_array(ncp, local_buff);
    CHECK_ERROR(err);
  // Communicate the sizes of the header structure for each process
    MPI_Offset* all_collection_sizes = (MPI_Offset*) NCI_Malloc(nproc * sizeof(MPI_Offset));
    TRACE_COMM(MPI_Allgather)(&local_buff_size, 1, MPI_OFFSET, all_collection_sizes, 1, MPI_OFFSET, ncp->comm);
    // if (rank == 1) printf("\nbefore comm rank %d: ncp->blocks.value[0]->vars.ndefined: %d\n", rank, ncp->blocks.value[0]->vars.ndefined);

    // Calculate displacements for the second phase
    int* recv_displs = (int*) NCI_Malloc(nproc * sizeof(int));
    int total_recv_size = all_collection_sizes[0];
    recv_displs[0] = 0;
    for (int i = 1; i < nproc; ++i) {
        recv_displs[i] = recv_displs[i - 1] + all_collection_sizes[i - 1];
        total_recv_size += all_collection_sizes[i];
        
    }
    char* all_collections_buffer = (char*) NCI_Malloc(total_recv_size);

    int* recvcounts =  (int*)NCI_Malloc(nproc * sizeof(int));
    for (int i = 0; i < nproc; ++i) {
        recvcounts[i] = (int)all_collection_sizes[i];
    }
    // if (rank == 1) printf("\nbefore comm rank %d: ncp->blocks.value[0]->vars.ndefined: %d\n", rank, ncp->blocks.value[0]->vars.ndefined);

    TRACE_COMM(MPI_Allgatherv)(local_buff, local_buff_size, MPI_BYTE, all_collections_buffer, recvcounts, recv_displs, MPI_BYTE, ncp->comm);

    //STEP3: adjust local block array based on the newly collected blocks
    int new_total = total_new_blocks + ncp->blocks.nread;
    ncp->blocks.value = (NC_block**) NCI_Realloc(ncp->blocks.value, new_total * sizeof(NC_block*));
    ncp->blocks.localids = (int*) NCI_Realloc(ncp->blocks.localids, new_total * X_SIZEOF_INT);
    ncp->blocks.globalids = (int*) NCI_Realloc(ncp->blocks.globalids, new_total * X_SIZEOF_INT);
    
    /*
    example: rank0 ABCDE|FG rank1 ABCDE|H  rank2 ABCDE|I ->   ABCDE|FGHI
    after merging rank0 globalids should be 01234|5678 rank1 should be 01234|7568
    */
    // if (rank == 1) printf("\nbefore comm rank %d: ncp->blocks.value[0]->vars.ndefined: %d\n", rank, ncp->blocks.value[0]->vars.ndefined);
    for(int i=ncp->blocks.ndefined; i<new_total;i++){
        ncp->blocks.value[i] = (NC_block*) NCI_Malloc(sizeof(NC_block));
    }

    for(int i=ncp->blocks.nread; i<ncp->blocks.ndefined;i++){
        if (new_block_offsets[rank] > 0){
            ncp->blocks.value[i + new_block_offsets[rank]] = ncp->blocks.value[i];
            ncp->blocks.value[i] = NULL;
        }
        ncp->blocks.globalids[i] = i +  new_block_offsets[rank];
    }
    //rank1 globalids is now 01234|7 _ _ _
    for(int i=0;i<new_block_offsets[rank];i++){
        ncp->blocks.globalids[ncp->blocks.nread + all_num_news[rank] + i] = ncp->blocks.nread + i;
    }
    //rank1 globalids is now 01234|7 5 6 _
    for(int i=ncp->blocks.nread +  + all_num_news[rank] + new_block_offsets[rank]; i<new_total;i++){
        ncp->blocks.globalids[i] = i;
    }
    //rank1 globalids is now 01234|7 5 6 8
    //Update localid accordingly
    for(int i=0; i<new_total;i++){
        ncp->blocks.localids[ncp->blocks.globalids[i]] = i;
    }


    //STEP4: Deseralize buffer to global block array (only name and block size info)
    err = deserialize_bufferinfo_array(ncp, all_collections_buffer, recv_displs, new_block_offsets, total_recv_size, nproc, rank);
    // if (rank == 1) printf("\nbefore comm rank %d: ncp->blocks.value[0]->vars.ndefined: %d\n", rank, ncp->blocks.value[0]->vars.ndefined);

    ncp->blocks.ndefined = new_total;
    // printf("\nlast:ncp->blocks.value[0]->name: %s\n", ncp->blocks.value[0]->name);
    // printf("\nlast:ncp->blocks.value[0]->dims.value[0]->name: %s\n", ncp->blocks.value[0]->dims.value[0]->name);

    CHECK_ERROR(err);
    NCI_Free(local_buff);
    NCI_Free(all_num_news);
    NCI_Free(all_collection_sizes);
    NCI_Free(recv_displs);
    NCI_Free(recvcounts);
    NCI_Free(all_collections_buffer);
    NCI_Free(new_block_offsets);
    err = NC_begins(ncp);
    CHECK_ERROR(err)
    /* update the total number of record variables */
    ncp->vars.num_rec_vars = 0;
    for (i=0; i<ncp->vars.ndefined; i++)
        ncp->vars.num_rec_vars += IS_RECVAR(ncp->vars.value[i]);

    if (ncp->safe_mode) {
        /* check whether variable begins are in an increasing order.
         * This check is for debugging purpose. */
        err = ncmpio_NC_check_voffs(ncp);
        CHECK_ERROR(err)
    }

#ifdef ENABLE_SUBFILING
    if (ncp->num_subfiles > 1) {
        /* get ncp info for the subfile */
        err = NC_begins(ncp->ncp_sf);
        CHECK_ERROR(err)

        if (ncp->safe_mode) {
            /* check whether variable begins are in an increasing order.
             * This check is for debugging purpose. */
            err = ncmpio_NC_check_voffs(ncp->ncp_sf);
            CHECK_ERROR(err)
        }
    }
#endif
    //META: TODO: need to modify the following code to handle redef() case
    if (ncp->old != NULL) {
        /* The current define mode was entered from ncmpi_redef, not from
         * ncmpi_create. We must check if header has been expanded.
         */

        assert(!NC_IsNew(ncp));
        assert(fIsSet(ncp->flags, NC_MODE_DEF));
        assert(ncp->begin_rec >= ncp->old->begin_rec);
        assert(ncp->begin_var >= ncp->old->begin_var);
        assert(ncp->vars.ndefined >= ncp->old->vars.ndefined);
        /* ncp->numrecs has already sync-ed in ncmpi_redef */

        if (ncp->vars.ndefined > 0) { /* no. record and non-record variables */
            if (ncp->begin_var > ncp->old->begin_var) {
                /* header size increases, shift the entire data part down */
                /* shift record variables first */
                err = move_record_vars(ncp, ncp->old);
                CHECK_ERROR(err)

                /* shift non-record variables */
                /* err = move_vars_r(ncp, ncp->old); */
                err = move_fixed_vars(ncp, ncp->old);
                CHECK_ERROR(err)
            }
            else if (ncp->begin_rec > ncp->old->begin_rec ||
                     ncp->recsize   > ncp->old->recsize) {
                /* number of non-record variables increases, or
                   number of records of record variables increases,
                   shift and move all record variables down */
                err = move_record_vars(ncp, ncp->old);
                CHECK_ERROR(err)
            }
        }
    } /* ... ncp->old != NULL */

    /* first sync header objects in memory across all processes, and then root
     * writes the header to file. Note safe_mode error check will be done in
     * write_NC() */
    // if (rank == 1) printf("\nbefore write NC ncp->global_xsz: %lld", ncp->global_xsz);
    status = write_NC(ncp);

    /* we should continue to exit define mode, even if header is inconsistent
     * among processes, so the program can proceed, say to close file properly.
     * However, if ErrIsHeaderDiff(status) is true, this error should
     * be considered fatal, as inconsistency is about the data structure,
     * rather then contents (such as attribute values) */

#ifdef ENABLE_SUBFILING
    /* write header to subfile */
    if (ncp->num_subfiles > 1) {
        err = write_NC(ncp->ncp_sf);
        if (status == NC_NOERR) status = err;
    }
#endif

    /* fill variables according to their fill mode settings */
    if (ncp->vars.ndefined > 0) {
        err = ncmpio_fill_vars(ncp);
        if (status == NC_NOERR) status = err;
    }

    if (ncp->old != NULL) {
        ncmpio_free_NC(ncp->old);
        ncp->old = NULL;
    }
    fClr(ncp->flags, NC_MODE_CREATE | NC_MODE_DEF);

#ifdef ENABLE_SUBFILING
    if (ncp->num_subfiles > 1)
        fClr(ncp->ncp_sf->flags, NC_MODE_CREATE | NC_MODE_DEF);
#endif
   
    return status;
}

/*----< ncmpio_enddef() >----------------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpio_enddef(void *ncdp)
{
    return ncmpio__enddef(ncdp, 0, 0, 0, 0);
}

