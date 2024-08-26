/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the corresponding APIs defined in
 * src/dispatchers/file.c
 *
 * ncmpi_def_block()    : dispatcher->def_block()
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <mpi.h>
#include <pnetcdf.h>
#include <pnc_debug.h>
#include <common.h>
#include "ncmpio_NC.h"
#include <ncx.h>

/*----< ncmpio_def_block() >---------------------------------------------------*/
int
ncmpio_def_block(void       *ncdp,    /* IN:  NC object */
               const char *name,    /* IN:  name of block */
               int        *blkidp)  /* OUT: block ID */
{
    int blkid, err=NC_NOERR;
    char *nname=NULL;  /* normalized name */
    NC *ncp=(NC*)ncdp;
    NC_block *blockp=NULL;

    /* create a normalized character string */
    err = ncmpii_utf8_normalize(name, &nname);
    if (err != NC_NOERR) return err;

    /* create a new block object (blkp->name points to nname) */
    blockp = (NC_block*) NCI_Malloc(sizeof(NC_block));
    if (blockp == NULL) {
        NCI_Free(nname);
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }
    blockp->name     = nname;
    blockp->name_len = strlen(nname);
    blockp->modified = 1; /* set to modified */

        /* initialize unlimited_id as no unlimited dimension yet defined */
    blockp->dims.unlimited_id = -1;

    /* allocate/expand ncp->blocks.value array */
    if (ncp->blocks.ndefined % PNC_ARRAY_GROWBY == 0) {
        size_t alloc_size = (size_t)ncp->blocks.ndefined + PNC_ARRAY_GROWBY;

        ncp->blocks.value = (NC_block **) NCI_Realloc(ncp->blocks.value,
                                      alloc_size * sizeof(NC_block*));
        ncp->blocks.globalids = (int *) NCI_Realloc(ncp->blocks.globalids,
                                      alloc_size * sizeof(int));
        ncp->blocks.localids = (int *) NCI_Realloc(ncp->blocks.localids,
                                      alloc_size * sizeof(int));
        if (ncp->blocks.value == NULL) {
            NCI_Free(nname);
            NCI_Free(blockp);
            DEBUG_RETURN_ERROR(NC_ENOMEM)
        }
    }
    
    blkid = ncp->blocks.ndefined;
    /* Add a new blk handle to the end of handle array */
    ncp->blocks.globalids[blkid] = ncp->blocks.localids[blkid] = blkid;
    ncp->blocks.value[blkid] = blockp;
    ncp->blocks.ndefined++;
    /*init dim and var arrays*/

    ncp->blocks.value[blkid]->dims.ndefined = 0;
    ncp->blocks.value[blkid]->dims.value = NULL;
    ncp->blocks.value[blkid]->dims.nameT = NULL;
    ncp->blocks.value[blkid]->dims.hash_size = ncp->hash_size_dim;
    ncp->blocks.value[blkid]->vars.ndefined = 0;
    ncp->blocks.value[blkid]->vars.value = NULL;
    ncp->blocks.value[blkid]->vars.nameT = NULL;
    ncp->blocks.value[blkid]->vars.hash_size = ncp->hash_size_var;
    ncp->blocks.value[blkid]->block_var_len = 0;


#ifndef SEARCH_NAME_LINEARLY
    /* allocate hashing lookup table, if not allocated yet */
    if (ncp->blocks.nameT == NULL)
        ncp->blocks.nameT = NCI_Calloc(ncp->blocks.hash_size, sizeof(NC_nametable));

    /* insert nname to the lookup table */
    ncmpio_hash_insert(ncp->blocks.nameT, ncp->blocks.hash_size, nname, blkid);
#endif

    if (blkidp != NULL) *blkidp = blkid;

    return err;
}

/*----< NC_findblk() >-------------------------------------------------------*/
/*
 * Search name from hash table ncap->nameT.
 * If found, set the blk ID pointed by blkidp, otherwise return NC_EBADBLK
 */
static int
NC_findblk(const NC_blockarray *ncap,
           const char        *name,  /* normalized blk name */
           int               *blkidp)
{
    int i, key, blkid;
    size_t nchars;

    if (ncap->ndefined == 0)
        DEBUG_RETURN_ERROR(NC_EBADBLK)

    /* hash the blk name into a key for name lookup */
    key = HASH_FUNC(name, ncap->hash_size);

    /* check the list using linear search */
    nchars = strlen(name);
    for (i=0; i<ncap->nameT[key].num; i++) {
        blkid = ncap->nameT[key].list[i];
        if (ncap->value[blkid]->name_len == nchars &&
            strcmp(name, ncap->value[blkid]->name) == 0) {
            if (blkidp != NULL) *blkidp = blkid;
            return NC_NOERR; /* the name already exists */
        }
    }
    DEBUG_RETURN_ERROR(NC_EBADBLK) /* the name has never been used */
}

/*----< ncmpio_inq_blkid() >-------------------------------------------------*/
int
ncmpio_inq_blkid(void       *ncdp,
                 const char *name,
                 int        *blkid)
{
    int err=NC_NOERR;
    char *nname=NULL; /* normalized name */
    NC *ncp=(NC*)ncdp;

    /* create a normalized character string */
    err = ncmpii_utf8_normalize(name, &nname);
    if (err != NC_NOERR) return err;

    err = NC_findblk(&ncp->blocks, nname, blkid);
    NCI_Free(nname);

    return err;
}
/*----< ncmpio_free_NC_block() >-----------------------------------------*/
/* Free NC_block values. */
void
ncmpio_free_NC_block(NC_block *ncap)
{
    if (ncap == NULL) return;
    
    NCI_Free(ncap->name);
    ncmpio_free_NC_dimarray(&ncap->dims);
    ncmpio_free_NC_vararray(&ncap->vars);
}

/*----< ncmpio_free_NC_blockarray() >-----------------------------------------*/
/* Free NC_blockarray values. */
void
ncmpio_free_NC_blockarray(NC_blockarray *ncap)
{
    int i;

    assert(ncap != NULL);

    if (ncap->value != NULL) {
        /* when error is detected reading NC_ATTRIBUTE tag, ncap->ndefined can
         * be > 0 and ncap->value is still NULL
         */

        for (i=0; i<ncap->ndefined; i++) {
            if (ncap->value[i] == NULL) continue;
            ncmpio_free_NC_block(ncap->value[i]);
            NCI_Free(ncap->value[i]);
        }

        NCI_Free(ncap->value);
        ncap->value = NULL;
    }
    ncap->ndefined = 0;

#ifndef SEARCH_NAME_LINEARLY
    if (ncap->nameT != NULL) {
        ncmpio_hash_table_free(ncap->nameT, ncap->hash_size);
        NCI_Free(ncap->nameT);
        ncap->nameT = NULL;
        ncap->hash_size = 0;
    }
#endif
}


