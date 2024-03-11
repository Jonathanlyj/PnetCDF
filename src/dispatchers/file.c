/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>     /* getenv() */
#include <string.h>     /* strtok(), strtok_r(), strchr(), strcpy(), strdup() */
#include <strings.h>    /* strcasecmp() */
#include <fcntl.h>      /* open() */
#include <sys/types.h>  /* lseek() */
#include <unistd.h>     /* read(), close(), lseek() */
#include <assert.h>     /* assert() */
#include <errno.h>      /* errno */
#include "baseline_ncx.h" 
#include "../drivers/ncmpio/ncmpio_NC.h"

#ifdef ENABLE_THREAD_SAFE
#include<pthread.h>
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
#endif

#ifdef ENABLE_NETCDF4
#include <netcdf.h>
#endif

#include <pnetcdf.h>
#include <dispatch.h>
#include <pnc_debug.h>
#include <common.h>

#ifdef ENABLE_ADIOS
#include "adios_read.h"
#include <arpa/inet.h>
#define BP_MINIFOOTER_SIZE 28
#define BUFREAD64(buf,var) memcpy(&var, buf, 8); if (diff_endian) swap_64(&var);
#endif

/* Note accessing the following 3 global variables must be protected by a
 * mutex, otherwise it will not be thread safe.
 */

/* static variables are initialized to NULLs */
static PNC *pnc_filelist[NC_MAX_NFILES];
static int  pnc_numfiles;

/* This is the default create format for ncmpi_create and nc__create.
 * The use of this file scope variable is not thread-safe.
 */
static int ncmpi_default_create_format = NC_FORMAT_CLASSIC;

#define NCMPII_HANDLE_ERROR(func)                                  \
    if (mpireturn != MPI_SUCCESS) {                                \
        int errorStringLen;                                        \
        char errorString[MPI_MAX_ERROR_STRING];                    \
        MPI_Error_string(mpireturn, errorString, &errorStringLen); \
        printf("Error: file %s line %d calling func %s: (%s)\n",   \
               __FILE__, __LINE__, func, errorString);             \
    }

#define CHECK_ERRNO(err, func) {                                   \
    if (err != 0) {                                                \
        printf("Error: file %s line %d calling func %s: (%s)\n",   \
               __FILE__, __LINE__, func, strerror(err));           \
        goto err_out;                                              \
    }                                                              \
}
/*META: helper function*/
static int
NC_finddim_helper(const NC_dimarray *ncap,
           const char        *name,  /* normalized dim name */
           int               *dimidp)
{
    int i, key, dimid;
    size_t nchars;

    if (ncap->ndefined == 0) return NC_EBADDIM;

    /* hash the dim name into a key for name lookup */
    key = HASH_FUNC(name);

    /* check the list using linear search */
    nchars = strlen(name);
    for (i=0; i<ncap->nameT[key].num; i++) {
        dimid = ncap->nameT[key].list[i];
        if (ncap->value[dimid]->name_len == nchars &&
            strcmp(name, ncap->value[dimid]->name) == 0) {
            if (dimidp != NULL) *dimidp = dimid;
            return NC_NOERR; /* the name already exists */
        }
    }
    return NC_EBADDIM; /* the name has never been used */
}

/*META: Extract metadata and save it to new header struc*/

static int baseline_extract_meta(void *ncdp, struct hdr *file_info) {

    int ncid, num_vars, num_dims, tot_num_dims, elem_sz, v_attrV_xsz;
    int err = NC_NOERR;
    MPI_Offset start, count;
    NC *ncp = (NC*)ncdp;
    
    file_info->vars.ndefined = ncp->vars.ndefined;
    file_info->xsz = 0;
    // Dimensions
    file_info->dims.ndefined = ncp->dims.ndefined;
    file_info->dims.value = (hdr_dim **)NCI_Malloc(file_info->dims.ndefined * sizeof(hdr_dim *));
    file_info->xsz += 2 * sizeof(uint32_t); // NC_Dimension and nelems

    for (int i = 0; i < file_info->dims.ndefined; i++) {
        hdr_dim *dim_info = (hdr_dim *)NCI_Malloc(sizeof(hdr_dim));
        dim_info->size = ncp->dims.value[i]->size;
        dim_info->name_len =  ncp->dims.value[i]->name_len;
        dim_info->name = (char *)NCI_Malloc(dim_info->name_len + 1);
        // dim_info->shared = false;
        // dim_info->global_idx = 0;
        strcpy(dim_info->name, ncp->dims.value[i]->name);
        file_info->dims.value[i] = dim_info;
        file_info->xsz += sizeof(uint32_t) + sizeof(char) * dim_info->name_len; // dim name
        file_info->xsz += sizeof(uint32_t); //size
    }
    // Variables
    file_info->vars.ndefined = ncp->vars.ndefined; 
    file_info->vars.value = (hdr_var **)NCI_Malloc(file_info->vars.ndefined * sizeof(hdr_var *));
    file_info->xsz += 2 * sizeof(uint32_t); // NC_Variable and ndefined
    for (int i = 0; i < file_info->vars.ndefined; i++) {
       hdr_var *var_info = (hdr_var *)NCI_Malloc(sizeof(hdr_var));
       var_info->xtype = ncp->vars.value[i]->xtype;
       var_info->name_len = ncp->vars.value[i]->name_len;
       var_info->name = (char *)NCI_Malloc(var_info->name_len + 1);
       strcpy(var_info->name, ncp->vars.value[i]->name);
       var_info->ndims = ncp->vars.value[i]->ndims;
       var_info->dimids = (int *)NCI_Malloc((i + 1) * sizeof(int));
        for (int j = 0; j <=var_info->ndims; j++) {
           var_info->dimids[j] = ncp->vars.value[i]->dimids[j];
        }
        // file_info->vars.value[i] = var_info;
        file_info->xsz += sizeof(uint32_t) + sizeof(char) *var_info->name_len; //var name
        file_info->xsz += sizeof(uint32_t); //xtype
        file_info->xsz += sizeof(uint32_t); //nelems of dim list
        file_info->xsz += sizeof(uint32_t) *var_info->ndims; // dimid list

        //Variable Attributes
        var_info->attrs.ndefined = ncp->vars.value[i]->attrs.ndefined;
        file_info->xsz += 2 * sizeof(uint32_t); // NC_Attribute and ndefine
        var_info->attrs.value = (hdr_attr **)NCI_Malloc(var_info->attrs.ndefined * sizeof(hdr_attr *));
        for (int k = 0; k < ncp->vars.value[i]->attrs.ndefined; k++) {
    
                    hdr_attr *attr_info = (hdr_attr *)NCI_Malloc(sizeof(hdr_attr));
                    attr_info->nelems = ncp->vars.value[i]->attrs.value[k]->nelems;
                    attr_info->xtype = ncp->vars.value[i]->attrs.value[k] ->xtype; // Using NC_INT for simplicity
                    attr_info->name_len = ncp->vars.value[i]->attrs.value[k]->name_len;
                    attr_info->name = (char *)NCI_Malloc(attr_info->name_len + 1);
                    strcpy(attr_info->name, ncp->vars.value[i]->attrs.value[k]->name);
                    // ncmpii_xlen_nc_type(attr_info->xtype, &v_attrV_xsz);
                    // int nbytes = attr_info->nelems * v_attrV_xsz;
                    // memcpy(attr_info->xvalue, ncp->vars.value[i]->attrs.value[k]->xvalue, nbytes);
                    attr_info->xvalue = ncp->vars.value[i]->attrs.value[k]->xvalue;
                    file_info->xsz += sizeof(uint32_t) + sizeof(char) * attr_info->name_len; //attr name
                    file_info->xsz += sizeof(uint32_t); // nc_type
                    file_info->xsz += sizeof(uint32_t); // nelems
                    err = xlen_nc_type(attr_info->xtype, &v_attrV_xsz);
                    file_info->xsz += attr_info->nelems * v_attrV_xsz;
        
                    var_info->attrs.value[k] = attr_info;
            }
        file_info->vars.value[i] = var_info;

        }


    return err;
}

/*META: Add metadata to header object*/
static int add_dim(NC* ncp, int *dimidp, char* name, MPI_Offset size){
    int dimid, err;
    char *nname=NULL;  /* normalized name */
    NC_dim *dimp=NULL;
    /* create a normalized character string */
    err = ncmpii_utf8_normalize(name, &nname);
    if (err != NC_NOERR) return err;

    /* create a new dimension object (dimp->name points to nname) */
    dimp = (NC_dim*) NCI_Malloc(sizeof(NC_dim));
    if (dimp == NULL) {
        NCI_Free(nname);
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    dimp->size     = size;
    dimp->name     = nname;
    dimp->name_len = strlen(nname);
  
    dimid = ncp->dims.ndefined;
    (*dimidp) = dimid;
    /* Add a new dim handle to the end of handle array */
    ncp->dims.value[dimid] = dimp;
    //TODO: check unlimited id conflicts
    if (dimp->size == NC_UNLIMITED) ncp->dims.unlimited_id = dimid;
    ncp->dims.ndefined++;

    #ifndef SEARCH_NAME_LINEARLY
        // ncmpio_hash_insert(ncp->dims.nameT, nname, dimid);
        ncmpio_hash_table_populate_NC_dim(&ncp->dims);
    #endif

    return NC_NOERR;

}

/*META: Add metadata to header object*/
static int add_hdr(struct hdr *hdr_data, int hdr_idx, int rank, PNC* pncp, const NC_dimarray* old_dimarray, const NC_vararray* old_vararray){
    // NC_dimarray* ncdims, NC_vararray* ncvars
    NC *ncp=(NC*)pncp->ncp;
    //add dimensions 
    int ndims= hdr_data->dims.ndefined;
    int cum_ndims = ncp->dims.ndefined;

    int i,j,k,nerrs=0;
    MPI_Offset len;
    int  dimid,err;
    int *new_indexes = (int*)NCI_Malloc(sizeof(int) * ndims);//mapping from old index to new index
    // check if total number of dimensions exceed max number allowed
    int tmp = cum_ndims + ndims;
    
    if(hdr_idx>0) tmp = cum_ndims + (ndims - old_dimarray->nread);
    if (tmp > NC_MAX_DIMS) DEBUG_RETURN_ERROR(NC_EMAXDIMS)
    //expand dimarray size
    size_t extra_chunk =  _RNDUP(tmp, NC_ARRAY_GROWBY) - _RNDUP(cum_ndims, NC_ARRAY_GROWBY);
    // printf("\ntmp is %d, cum_ndims is %d, ndims is %d", tmp, cum_ndims, ndims); 

    if (extra_chunk > 0){
        size_t alloc_size = (size_t)_RNDUP(tmp, NC_ARRAY_GROWBY);
        
        ncp->dims.value = (NC_dim **) NCI_Realloc(ncp->dims.value,
                                      alloc_size * sizeof(NC_dim*));
        ncp->dims.localids = (int *) NCI_Realloc(ncp->dims.localids,
                                      alloc_size * SIZEOF_INT);
        ncp->dims.indexes = (int *) NCI_Realloc(ncp->dims.indexes,
                                    alloc_size * SIZEOF_INT);
        if (ncp->dims.value == NULL || ncp->dims.localids == NULL || ncp->dims.indexes == NULL)
            DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    //new local id generator
    int dimid_generator,varid_generator;
    dimid_generator =  ncp->dims.ndefined;
    varid_generator =  ncp->vars.ndefined;
    // if(hdr_idx<rank){
    //     dimid_generator = ncp->dims.ndefined + old_dimarray->ndefined;
    //     varid_generator = ncp->vars.ndefined + old_vararray->ndefined;
    // }else{
    //     dimid_generator = ncp->dims.ndefined;
    //     varid_generator = ncp->vars.ndefined;
    // }

    //store dims
        /*Reorganize this:
            if (hdr_idx == 0) and (i < nread): //intial definition for dim read from file
                define as new dimension;
                maintain old local ids from old dimarray;
                update local-global index mapping (for variable)
            elif(hdr_idx > 0) and (i < nread):
                already defined; skip define;
                maintain old local ids from old dimarray;
                update local-global index mapping (for variable)
            else
                check if name match:
                if name match:
                    retrieve dimid
                    check if info match
                    if info match: //existed dim no need to define
                        pdate local-global index mapping (for variable)
                        if rank == hdr_idx:
                            maintain old local ids:
                            localids[dimid] = old_dimarray.localids[i]
                            correct localids previously occupied
                            localids[old_dimarray.localids[i]] = dimid_generator++;
                        //other processes no need to change localuds at all
                    else: //info not match
                        error out
                        
                else: //no duplicated name
                    define as a new dimension;
                    update local-global index mapping (for variable)
                    if rank==hdr_idx:
                        maintain old local ids:
                        localids[dimid] = old_dimarray.localids[i]
                        correct localids previously occupied
                        localids[old_dimarray.localids[i]] = dimid_generator++;
                    else:
                        add local ids using id from id generater
                        
        */

     for (i=0; i<ndims; i++){
        // int shared_dim = 0;

       if(i < old_dimarray->nread){
            //intial definition for dim read from file
            
            if (hdr_idx == 0){
               
                err = add_dim(ncp, &dimid, hdr_data->dims.value[i]->name,hdr_data->dims.value[i]->size);
                if (err != NC_NOERR) return err;
                
                //update pnc header
                pncp->ndims++;
                if (hdr_data->dims.value[i]->size == NC_UNLIMITED && pncp->unlimdimid == -1) pncp->unlimdimid = dimid;
            }
            //maintain the old index to localid mapping       
            ncp->dims.localids[i] = old_dimarray->localids[i];
            //update local-global index mapping (for variable)
            new_indexes[i] = i;
        }else{ //newly defined dims
            
            err = pncp->driver->inq_dimid(ncp, hdr_data->dims.value[i]->name, &dimid);
            if (err != NC_EBADDIM) {
                //name matched, check property
                len = ncp->dims.value[dimid]->size;
                if (len!= hdr_data->dims.value[i]->size){
                    //duplicated name but different value error out
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_DIM_NAME)
                    goto err_check;
                }else{
                    //a shread dimension, skip define since its read from file and already defined
                    
                    if (rank == hdr_idx){
                        //maintain old local id for this dim
                        int prev_localid = ncp->dims.localids[dimid];
                        ncp->dims.localids[dimid] = old_dimarray->localids[i];
                        //correct the local id that was previously assigned to a dim
                        ncp->dims.localids[old_dimarray->localids[i]] = prev_localid;
                    }
                    //other processes has no need to change localids at all
                }
            }else{
                //new dimension, no duplicated name

                err = add_dim(ncp, &dimid, hdr_data->dims.value[i]->name,hdr_data->dims.value[i]->size);
                if (err != NC_NOERR) goto err_check;
                pncp->ndims++;
                
                if (hdr_data->dims.value[i]->size == NC_UNLIMITED && pncp->unlimdimid == -1) pncp->unlimdimid = dimid;
                new_indexes[i] = dimid;
                
                if (rank == hdr_idx){
                        //maintain old local id for this dim
                        ncp->dims.localids[dimid] = old_dimarray->localids[i];
                        //correct the local id that was previously assigned to a dim
                        ncp->dims.localids[old_dimarray->localids[i]] = dimid_generator++;
                    }else{// add local ids using id from id generater
                        ncp->dims.localids[dimid] = dimid_generator++;
                    }
            }

        }
     }     

    //add variables
    int nvars = hdr_data->vars.ndefined;
    int cum_nvars = ncp->vars.ndefined;
    // int *varid = (int *)malloc(nvars * sizeof(int));
    int v_ndims, v_namelen, xtype, n_att, varid;
    int *v_dimids;
    tmp = nvars + cum_nvars;
    if(hdr_idx>0) tmp = cum_ndims + (ndims - old_vararray->nread);
    //Expand NC object vararray
    extra_chunk =  _RNDUP(tmp, NC_ARRAY_GROWBY) - _RNDUP(cum_nvars, NC_ARRAY_GROWBY);

    if (extra_chunk > 0){
        size_t alloc_size = (size_t) _RNDUP(tmp, NC_ARRAY_GROWBY);
        
        ncp->vars.value = (NC_var **) NCI_Realloc(ncp->vars.value, alloc_size * sizeof(NC_var*));
        ncp->vars.localids = (int *) NCI_Realloc(ncp->vars.localids,
                                      alloc_size * SIZEOF_INT);
        ncp->vars.indexes = (int *) NCI_Realloc(ncp->vars.indexes,
                                    alloc_size * SIZEOF_INT);

        if (ncp->vars.value == NULL)
            DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    //Expand PNC object vararray

    extra_chunk =  _RNDUP(tmp, PNC_VARS_CHUNK) - _RNDUP(cum_nvars, PNC_VARS_CHUNK);
    if (extra_chunk > 0){
        size_t alloc_size = (size_t) _RNDUP(tmp, PNC_VARS_CHUNK);
        pncp->vars = NCI_Realloc(pncp->vars,
                                 alloc_size *sizeof(PNC_var));
    }

    // printf("\nnvars: %d", nvars);
    //store vars
        /*Reorganize this:
            if hdr_idx == 0:
                define as new variable
                if i < nread:
                    maintain the old index to localid mapping
                else:
                    new variable, create new id mapping
            else:
                if i < nread:
                    skip because already defined
                else:
                    if name matching, error out
                    if not, define as new variable create new mapping for localid;
        */

    for (i=0; i<nvars; i++){
        if(hdr_idx > 0){
            if (i < old_vararray->nread){
                continue;
            }else{
                //check if name already used
                err = pncp->driver->inq_varid(pncp->ncp, hdr_data->vars.value[i]->name, &varid);
                if (err != NC_ENOTVAR){
                //Disable shared variable across processes for now
                DEBUG_ASSIGN_ERROR(err, NC_ENAMEINUSE)
                goto err_check;
                }
            }
        }
        //All other cases: create the new variable
        v_namelen = hdr_data->vars.value[i]->name_len;
        xtype = hdr_data->vars.value[i]->xtype;
        v_ndims = hdr_data->vars.value[i]->ndims;

        char *nname=NULL;  /* normalized name */
        NC_var *varp=NULL;

        v_dimids = (int *)NCI_Malloc(v_ndims * sizeof(int));
        for(j=0; j<v_ndims; j++) v_dimids[j] = new_indexes[hdr_data->vars.value[i]->dimids[j]];

        /* create a normalized character string */
        err = ncmpii_utf8_normalize(hdr_data->vars.value[i]->name, &nname);
        if (err != NC_NOERR) goto err_check;
        //Add var to NC header object

        /* allocate a new NC_var object */
        varp = ncmpio_new_NC_var(nname, strlen(nname), v_ndims);

   
        varp->xtype = xtype;
        ncmpii_xlen_nc_type(xtype, &varp->xsz);
        /* copy dimids[] */
        if (v_ndims != 0 && v_dimids != NULL)
            memcpy(varp->dimids, v_dimids, (size_t)v_ndims * SIZEOF_INT);
        /* set up array dimensional structures */
        err = ncmpio_NC_var_shape64(varp, &ncp->dims);
        if (err != NC_NOERR) {
            ncmpio_free_NC_var(varp);
            nname = NULL; /* already freed in ncmpio_free_NC_var() */
            goto err_check;
        }
        /* Add a new dim handle to the end of handle array */

        varp->varid = ncp->vars.ndefined;
        ncp->vars.value[ncp->vars.ndefined] = varp;
        ncp->vars.ndefined++;
#ifndef SEARCH_NAME_LINEARLY
    /* insert nname to the lookup table */
        // ncmpio_hash_insert(ncp->vars.nameT, nname, varp->varid);
        ncmpio_hash_table_populate_NC_var(&ncp->vars);
        
#endif
        //Update PNC object
  
        /* default is NOFILL */
        varp->no_fill = 1;

        /* change to FILL only if the entire dataset fill mode is FILL */
        if (NC_dofill(ncp)) varp->no_fill = 0;

        //Add var to PNC object
        pncp->vars[varp->varid].ndims  = v_ndims;
            pncp->vars[varp->varid].xtype  = xtype;
            pncp->vars[varp->varid].recdim = -1;   /* if fixed-size variable */
            pncp->vars[varp->varid].shape  = NULL;
            if (ndims > 0) {
                if (v_dimids[0] == pncp->unlimdimid) { /* record variable */
                    pncp->vars[varp->varid].recdim = pncp->unlimdimid;
                    pncp->nrec_vars++;
                }

                pncp->vars[varp->varid].shape = (MPI_Offset*)
                                            NCI_Malloc(v_ndims * SIZEOF_MPI_OFFSET);
                for (int dim_i=0; dim_i<ndims; dim_i++) {
                    /* obtain size of dimension i */
                    err = pncp->driver->inq_dim(pncp->ncp, v_dimids[dim_i], NULL,
                                                pncp->vars[varp->varid].shape+dim_i);
                    if (err != NC_NOERR) return err;
                }
            }
            pncp->nvars++;

        // Add variable attributes

        int att_namelen, att_xtype, att_nelems,v_attr_xsz, nbytes;
        int nattrs = hdr_data->vars.value[i]->attrs.ndefined;
        int att_vid = varp->varid;
        // printf("\nvariable %d: nattrs: %d", i, nattrs);
        // int *varid = (int *)malloc(nattrs * sizeof(int));
        ncp->vars.value[att_vid]->attrs.ndefined = nattrs;
        size_t alloc_size = _RNDUP(nattrs, NC_ARRAY_GROWBY);
        alloc_size = 0;

        ncp->vars.value[att_vid]->attrs.value = (NC_attr**) NCI_Calloc(alloc_size, sizeof(NC_attr*));
        if (ncp->vars.value[att_vid]->attrs.value == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

        for(k=0; k<nattrs; k++){
            NC_attr *attrp=NULL;
            att_namelen = hdr_data->vars.value[i]->attrs.value[k]->name_len;
            att_xtype = hdr_data->vars.value[i]->attrs.value[k]->xtype;
            att_nelems = hdr_data->vars.value[i]->attrs.value[k]->nelems;
            
            /* create a normalized character string */
            err = ncmpii_utf8_normalize(hdr_data->vars.value[i]->attrs.value[k]->name, &nname);
            if (err != NC_NOERR) goto err_check;
            err = ncmpio_new_NC_attr(nname, att_namelen, att_xtype, att_nelems, &attrp);

            if (err != NC_NOERR) {
                NCI_Free(nname);
                return err;
            }
            ncmpii_xlen_nc_type(att_xtype, &v_attr_xsz);
            nbytes = attrp->nelems * v_attr_xsz;
            memcpy(attrp->xvalue, hdr_data->vars.value[i]->attrs.value[k]->xvalue, nbytes);
            ncp->vars.value[att_vid]->attrs.value[k] = attrp;
        }

        NCI_Free(v_dimids);
        //Map localids for all cases
        if (hdr_idx == 0 && i < old_vararray->nread){
            // variable read from file, all processes maintain their original
            ncp->vars.localids[i] = old_vararray->localids[i];
        }else if(rank == hdr_idx){
            // the "host" process need to maintain the origianl mapping
            
            ncp->vars.localids[varp->varid] = old_vararray->localids[i];
             //correct the local id that was previously assigned to a previous var
            // printf("rank %d: %d, %d, %d", rank, i, old_vararray->localids[i], varid_generator);
            ncp->vars.localids[old_vararray->localids[i]] = varid_generator++;
            
        }else{
            // a new var not seen by previous old vararray
            ncp->vars.localids[varp->varid] = varid_generator++;
            // printf("\nrank %d: %d, %d", rank,varp->varid, varid_generator);
        }


       
    }
    NCI_Free(new_indexes);

err_check:
    if (err != NC_NOERR) return err;
    return nerrs;
}


// /*META: Add dim metadata to header object*/
// static int add_dim_hdr(struct hdr *hdr_data, int hdr_idx, int rank, PNC* pncp){
//     NC *ncp=(NC*)pncp->ncp;
//     //add dimensions 
//     int ndims= hdr_data->dims.ndefined;
//     int cum_ndims = ncp->dims.ndefined;

//     int i,j,k,nerrs=0;
//     MPI_Offset len;
//     int  dimid,err;
//     int *new_indexes = (int*)NCI_Malloc(sizeof(int) * ndims);//mapping from old index to new index

// }


/*----< new_id_PNCList() >---------------------------------------------------*/
/* Return a new ID (array index) from the PNC list, pnc_filelist[] that is
 * not used. Note the used elements in pnc_filelist[] may not be contiguous.
 * For example, some files created/opened later may be closed earlier than
 * others, leaving those array elements NULL in the middle.
 */
static int
new_id_PNCList(int *new_id, PNC *pncp)
{
    int i, err=NC_NOERR, perr=0;

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_lock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_lock")
#endif
    *new_id = -1;
    if (pnc_numfiles == NC_MAX_NFILES) { /* Too many files open */
        DEBUG_ASSIGN_ERROR(err, NC_ENFILE)
    }
    else {
        err = NC_NOERR;
        for (i=0; i<NC_MAX_NFILES; i++) { /* find the first unused element */
            if (pnc_filelist[i] == NULL) {
                *new_id = i;
                pnc_filelist[i] = pncp;
                pnc_numfiles++; /* increment number of files opened */
                break;
            }
        }
    }
#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_unlock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_unlock")

err_out:
#endif
    return (err != NC_NOERR) ? err : perr;
}

/*----< del_from_PNCList() >-------------------------------------------------*/
static int
del_from_PNCList(int ncid)
{
    int perr=0;

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_lock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_lock")
#endif

    /* validity of ncid should have been checked already */
    pnc_filelist[ncid] = NULL;
    pnc_numfiles--;

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_unlock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_unlock")

err_out:
#endif
    return perr;
}

/*----< PNC_check_id() >-----------------------------------------------------*/
int
PNC_check_id(int ncid, PNC **pncp)
{
    int err=NC_NOERR, perr=0;

    assert(pncp != NULL);

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_lock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_lock")
#endif

    if (pnc_numfiles == 0 || ncid < 0 || ncid >= NC_MAX_NFILES)
        DEBUG_ASSIGN_ERROR(err, NC_EBADID)
    else
        *pncp = pnc_filelist[ncid];

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_unlock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_unlock")

err_out:
#endif
    return (err != NC_NOERR) ? err : perr;
}

/*----< construct_info() >---------------------------------------------------*/
static void
combine_env_hints(MPI_Info  user_info,  /* IN */
                  MPI_Info *new_info)   /* OUT: may be MPI_INFO_NULL */
{
    char *warn_str="Warning: skip ill-formed hint set in PNETCDF_HINTS";
    char *env_str;

    /* take hints from the environment variable PNETCDF_HINTS, a string of
     * hints separated by ";" and each hint is in the form of hint=value. E.g.
     * "cb_nodes=16;cb_config_list=*:6". If this environment variable is set,
     * it overrides the same hints that were set by MPI_Info_set() called in
     * the application program.
     */
    if (user_info != MPI_INFO_NULL)
        MPI_Info_dup(user_info, new_info); /* ignore error */
    else
        *new_info = MPI_INFO_NULL;

    /* get environment variable PNETCDF_HINTS */
    if ((env_str = getenv("PNETCDF_HINTS")) != NULL) {
#ifdef USE_STRTOK_R
        char *env_str_cpy, *env_str_saved, *hint, *key;
        env_str_cpy = strdup(env_str);
        env_str_saved = env_str_cpy;
        hint = strtok_r(env_str_cpy, ";", &env_str_saved);
        while (hint != NULL) {
            char *hint_saved = strdup(hint);
            char *val = strchr(hint, '=');
            if (val == NULL) { /* ill-formed hint */
                if (NULL != strtok(hint, " \t"))
                    printf("%s: '%s'\n", warn_str, hint_saved);
                /* else case: ignore white-spaced hints */
                free(hint_saved);
                hint = strtok_r(NULL, ";", &env_str_saved); /* get next hint */
                continue;
            }
            key = strtok(hint, "= \t");
            val = strtok(NULL, "= \t");
            if (NULL != strtok(NULL, "= \t")) /* expect no more token */
                printf("%s: '%s'\n", warn_str, hint_saved);
            else {
                if (*new_info == MPI_INFO_NULL)
                    MPI_Info_create(new_info); /* ignore error */
                MPI_Info_set(*new_info, key, val); /* override or add */
            }
            /* printf("env hint: key=%s val=%s\n",key,val); */
            hint = strtok_r(NULL, ";", &env_str_saved);
            free(hint_saved);
        }
        free(env_str_cpy);
#else
        char *env_str_cpy, *hint, *next_hint, *key, *val, *deli;
        char *hint_saved=NULL;

        env_str_cpy = strdup(env_str);
        next_hint = env_str_cpy;

        do {
            hint = next_hint;
            deli = strchr(hint, ';');
            if (deli != NULL) {
                *deli = '\0'; /* add terminate char */
                next_hint = deli + 1;
            }
            else next_hint = "\0";
            if (hint_saved != NULL) free(hint_saved);

            /* skip all-blank hint */
            hint_saved = strdup(hint);
            if (strtok(hint, " \t") == NULL) continue;

            free(hint_saved);
            hint_saved = strdup(hint); /* save hint for error message */

            deli = strchr(hint, '=');
            if (deli == NULL) { /* ill-formed hint */
                printf("%s: '%s'\n", warn_str, hint_saved);
                continue;
            }
            *deli = '\0';

            /* hint key */
            key = strtok(hint, "= \t");
            if (key == NULL || NULL != strtok(NULL, "= \t")) {
                /* expect one token before = */
                printf("%s: '%s'\n", warn_str, hint_saved);
                continue;
            }

            /* hint value */
            val = strtok(deli+1, "= \t");
            if (NULL != strtok(NULL, "= \t")) { /* expect one token before = */
                printf("%s: '%s'\n", warn_str, hint_saved);
                continue;
            }
            if (*new_info == MPI_INFO_NULL)
                MPI_Info_create(new_info); /* ignore error */
            MPI_Info_set(*new_info, key, val); /* override or add */

        } while (*next_hint != '\0');

        if (hint_saved != NULL) free(hint_saved);
        free(env_str_cpy);
#endif
    }
    /* return no error as all hints are advisory */
}

/*----< ncmpi_create() >-----------------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi_create(MPI_Comm    comm,
             const char *path,
             int         cmode,
             MPI_Info    info,
             int        *ncidp)
{
    int rank, nprocs, status=NC_NOERR, err;
    int safe_mode=0, mpireturn, relax_coord_bound, format;
    char *env_str;
    MPI_Info combined_info;
    void *ncp;
    PNC *pncp;
    PNC_driver *driver;
#ifdef BUILD_DRIVER_FOO
    int enable_foo_driver=0;
#endif
#ifdef ENABLE_BURST_BUFFER
    int enable_bb_driver=0;
#endif

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

#ifdef PNETCDF_DEBUG
    safe_mode = 1;
    /* When debug mode is enabled at the configure time, safe_mode is by
     * default enabled. This can be overwritten by the run-time environment
     * variable PNETCDF_SAFE_MODE */
#endif
    /* get environment variable PNETCDF_SAFE_MODE
     * if it is set to 1, then we perform a strict parameter consistent test
     */
    if ((env_str = getenv("PNETCDF_SAFE_MODE")) != NULL) {
        if (*env_str == '0') safe_mode = 0;
        else                 safe_mode = 1;
        /* if PNETCDF_SAFE_MODE is set but without a value, *env_str can
         * be '\0' (null character). In this case, safe_mode is enabled */
    }

    /* get environment variable PNETCDF_RELAX_COORD_BOUND
     * if it is set to 0, then we perform a strict start bound check
     */
#ifndef RELAX_COORD_BOUND
    relax_coord_bound = 0;
#else
    relax_coord_bound = 1;
#endif
    if ((env_str = getenv("PNETCDF_RELAX_COORD_BOUND")) != NULL) {
        if (*env_str == '0') relax_coord_bound = 0;
        else                 relax_coord_bound = 1;
        /* if PNETCDF_RELAX_COORD_BOUND is set but without a value, *env_str
         * can be '\0' (null character). This is equivalent to setting
         * relax_coord_bound to 1 */
    }

    /* path's validity is checked in MPI-IO with error code MPI_ERR_BAD_FILE
     * path consistency is checked in MPI-IO with error code MPI_ERR_NOT_SAME
     */
    if (path == NULL || *path == '\0') DEBUG_RETURN_ERROR(NC_EBAD_FILE)

    if (nprocs > 1) { /* Check cmode consistency */
        int root_cmode = cmode; /* only root's matters */
        TRACE_COMM(MPI_Bcast)(&root_cmode, 1, MPI_INT, 0, comm);
        NCMPII_HANDLE_ERROR("MPI_Bcast")

        /* Overwrite cmode with root's cmode */
        if (root_cmode != cmode) {
            cmode = root_cmode;
            DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_CMODE)
        }

        if (safe_mode) { /* sync status among all processes */
            err = status;
            TRACE_COMM(MPI_Allreduce)(&err, &status, 1, MPI_INT, MPI_MIN, comm);
            NCMPII_HANDLE_ERROR("MPI_Allreduce")
        }
        /* continue to use root's cmode to create the file, but will report
         * cmode inconsistency error, if there is any */
    }

    /* combine user's MPI info and PNETCDF_HINTS env variable */
    combine_env_hints(info, &combined_info);

#ifdef BUILD_DRIVER_FOO
    if (combined_info == MPI_INFO_NULL)
        MPI_Info_create(&combined_info);
    {
        char value[MPI_MAX_INFO_VAL];
        int flag;

        /* check if nc_foo_driver is enabled */
        MPI_Info_get(combined_info, "nc_foo_driver", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag && strcasecmp(value, "enable") == 0)
            enable_foo_driver = 1;
    }
#endif
#ifdef ENABLE_BURST_BUFFER
    if (combined_info == MPI_INFO_NULL)
        MPI_Info_create(&combined_info);
    {
        char value[MPI_MAX_INFO_VAL];
        int flag;

        /* check if nc_burst_buf is enabled */
        MPI_Info_get(combined_info, "nc_burst_buf", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag && strcasecmp(value, "enable") == 0)
            enable_bb_driver = 1;
    }
#endif

    /* Use environment variable and cmode to tell the file format
     * which is later used to select the right driver.
     */

#ifdef ENABLE_NETCDF4
    /* It is illegal to have NC_64BIT_OFFSET & NC_64BIT_DATA & NC_NETCDF4 */
    if ((cmode & (NC_64BIT_OFFSET|NC_NETCDF4)) ==
                 (NC_64BIT_OFFSET|NC_NETCDF4) ||
        (cmode & (NC_64BIT_DATA|NC_NETCDF4)) ==
                 (NC_64BIT_DATA|NC_NETCDF4)) {
        if (combined_info != MPI_INFO_NULL)
            MPI_Info_free(&combined_info);
        DEBUG_RETURN_ERROR(NC_EINVAL_CMODE)
    }
#else
    if (cmode & NC_NETCDF4) {
        if (combined_info != MPI_INFO_NULL)
            MPI_Info_free(&combined_info);
        DEBUG_RETURN_ERROR(NC_ENOTBUILT)
    }
#endif

    /* It is illegal to have both NC_64BIT_OFFSET & NC_64BIT_DATA */
    if ((cmode & (NC_64BIT_OFFSET|NC_64BIT_DATA)) ==
                 (NC_64BIT_OFFSET|NC_64BIT_DATA)) {
        if (combined_info != MPI_INFO_NULL)
            MPI_Info_free(&combined_info);
        DEBUG_RETURN_ERROR(NC_EINVAL_CMODE)
    }

    /* Check if cmode contains format specific flag */
    if (fIsSet(cmode, NC_64BIT_DATA))
        format = NC_FORMAT_CDF5;
    else if (fIsSet(cmode, NC_64BIT_OFFSET))
        format = NC_FORMAT_CDF2;
    else if (fIsSet(cmode, NC_NETCDF4)) {
        if (fIsSet(cmode, NC_CLASSIC_MODEL))
            format = NC_FORMAT_NETCDF4_CLASSIC;
        else
            format = NC_FORMAT_NETCDF4;
    }
    else if (fIsSet(cmode, NC_CLASSIC_MODEL))
        format = NC_FORMAT_CLASSIC;
    else {
        /* if no file format flag is set in cmode, use default */
        ncmpi_inq_default_format(&format);
        if (format == NC_FORMAT_CDF5)
            cmode |= NC_64BIT_DATA;
        else if (format == NC_FORMAT_CDF2)
            cmode |= NC_64BIT_OFFSET;
        else if (format == NC_FORMAT_NETCDF4)
            cmode |= NC_NETCDF4;
        else if (format == NC_FORMAT_NETCDF4_CLASSIC)
            cmode |= NC_NETCDF4 | NC_CLASSIC_MODEL;
    }

#ifdef ENABLE_NETCDF4
    if (format == NC_FORMAT_NETCDF4 || format == NC_FORMAT_NETCDF4_CLASSIC) {
        driver = nc4io_inq_driver();
#ifdef ENABLE_BURST_BUFFER
        /* Burst buffering does not support NetCDF-4 files yet.
         * If hint nc_burst_buf is enabled in combined_info, disable it.
         */
        if (enable_bb_driver == 1)
            MPI_Info_set(combined_info, "nc_burst_buf", "disable");
        enable_bb_driver = 0;
#endif
    }
    else
#endif
#ifdef BUILD_DRIVER_FOO
    if (enable_foo_driver)
        driver = ncfoo_inq_driver();
    else
#endif
#ifdef ENABLE_BURST_BUFFER
    if (enable_bb_driver)
        driver = ncbbio_inq_driver();
    else
#endif
        /* default is the driver built on top of MPI-IO */
        driver = ncmpio_inq_driver();

    /* allocate a new PNC object */
    pncp = (PNC*) NCI_Malloc(sizeof(PNC));
    if (pncp == NULL) {
        *ncidp = -1;
        if (combined_info != MPI_INFO_NULL)
            MPI_Info_free(&combined_info);
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    /* generate a new nc file ID from NCPList */
    err = new_id_PNCList(ncidp, pncp);
    if (err != NC_NOERR) {
        if (combined_info != MPI_INFO_NULL)
            MPI_Info_free(&combined_info);
        return err;
    }

    /* Duplicate comm, because users may free it (though unlikely). Note
     * MPI_Comm_dup() is collective. We pass pncp->comm to drivers, so there
     * is no need for a driver to duplicate it again.
     */
    if (comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF)
        MPI_Comm_dup(comm, &pncp->comm);
    else
        pncp->comm = comm;

    /* calling the driver's create subroutine */
    err = driver->create(pncp->comm, path, cmode, *ncidp, combined_info, &ncp);
    if (status == NC_NOERR) status = err;
    if (combined_info != MPI_INFO_NULL) MPI_Info_free(&combined_info);
    if (status != NC_NOERR && status != NC_EMULTIDEFINE_CMODE) {
        del_from_PNCList(*ncidp);
        if (pncp->comm != MPI_COMM_WORLD && pncp->comm != MPI_COMM_SELF)
            MPI_Comm_free(&pncp->comm); /* a collective call */
        NCI_Free(pncp);
        *ncidp = -1;
        return status;
    }

    /* fill in pncp members */
    pncp->path = (char*) NCI_Malloc(strlen(path)+1);
    if (pncp->path == NULL) {
        driver->close(ncp); /* close file and ignore error */
        del_from_PNCList(*ncidp);
        if (pncp->comm != MPI_COMM_WORLD && pncp->comm != MPI_COMM_SELF)
            MPI_Comm_free(&pncp->comm); /* a collective call */
        NCI_Free(pncp);
        *ncidp = -1;
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }
    strcpy(pncp->path, path);
    pncp->mode       = cmode;
    pncp->driver     = driver;
    pncp->ndims      = 0;
    pncp->unlimdimid = -1;
    pncp->nvars      = 0;
    pncp->nrec_vars  = 0;
    pncp->vars       = NULL;
    pncp->flag       = NC_MODE_DEF | NC_MODE_CREATE;
    pncp->ncp        = ncp;
    pncp->format     = format;

    if (safe_mode)          pncp->flag |= NC_MODE_SAFE;
    if (!relax_coord_bound) pncp->flag |= NC_MODE_STRICT_COORD_BOUND;

    return status;
}

#define _NDIMS_ 16

/*----< ncmpi_open() >-------------------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi_open(MPI_Comm    comm,
           const char *path,
           int         omode,
           MPI_Info    info,
           int        *ncidp)  /* OUT */
{
    int i, j, nalloc, rank, nprocs, format, status=NC_NOERR, err;
    int safe_mode=0, mpireturn, relax_coord_bound, DIMIDS[_NDIMS_], *dimids;
    char *env_str;
    MPI_Info combined_info;
    void *ncp;
    PNC *pncp;
    PNC_driver *driver;
#ifdef BUILD_DRIVER_FOO
    int enable_foo_driver=0;
#endif
#ifdef ENABLE_BURST_BUFFER
    int enable_bb_driver=0;
#endif

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

#ifdef PNETCDF_DEBUG
    safe_mode = 1;
    /* When debug mode is enabled at the configure time, safe_mode is by
     * default enabled. This can be overwritten by the run-time environment
     * variable PNETCDF_SAFE_MODE */
#endif
    /* get environment variable PNETCDF_SAFE_MODE
     * if it is set to 1, then we perform a strict parameter consistent test
     */
    if ((env_str = getenv("PNETCDF_SAFE_MODE")) != NULL) {
        if (*env_str == '0') safe_mode = 0;
        else                 safe_mode = 1;
        /* if PNETCDF_SAFE_MODE is set but without a value, *env_str can
         * be '\0' (null character). In this case, safe_mode is enabled */
    }

    /* get environment variable PNETCDF_RELAX_COORD_BOUND
     * if it is set to 0, then we perform a strict start bound check
     */
#ifndef RELAX_COORD_BOUND
    relax_coord_bound = 0;
#else
    relax_coord_bound = 1;
#endif
    if ((env_str = getenv("PNETCDF_RELAX_COORD_BOUND")) != NULL) {
        if (*env_str == '0') relax_coord_bound = 0;
        else                 relax_coord_bound = 1;
        /* if PNETCDF_RELAX_COORD_BOUND is set but without a value, *env_str
         * can be '\0' (null character). This is equivalent to setting
         * relax_coord_bound to 1 */
    }

    /* path's validity is checked in MPI-IO with error code MPI_ERR_BAD_FILE
     * path consistency is checked in MPI-IO with error code MPI_ERR_NOT_SAME
     */
    if (path == NULL || *path == '\0') DEBUG_RETURN_ERROR(NC_EBAD_FILE)

    /* Check the file signature to tell the file format which is later used to
     * select the right driver.
     */
    format = NC_FORMAT_UNKNOWN;
    if (rank == 0) {
        err = ncmpi_inq_file_format(path, &format);
        if (err != NC_NOERR) {
            if (nprocs == 1) return err;
            format = err;
        }
        else if (format == NC_FORMAT_UNKNOWN) {
            if (nprocs == 1) DEBUG_RETURN_ERROR(NC_ENOTNC)
            format = NC_ENOTNC;
        }
#ifndef ENABLE_NETCDF4
        else if (format == NC_FORMAT_NETCDF4 || format == NC_FORMAT_NETCDF4_CLASSIC) {
            if (nprocs == 1) DEBUG_RETURN_ERROR(NC_ENOTBUILT)
            format = NC_ENOTBUILT;
        }
#endif
    }

    if (nprocs > 1) { /* root broadcasts format and omode */
        int root_omode, msg[2];

        msg[0] = format; /* only root's matters (format or error code) */

        /* Check omode consistency:
         * Note if omode contains NC_NOWRITE, it is equivalent to NC_CLOBBER.
         * In pnetcdf.h, they both are defined the same value, 0.
         * Only root's omode matters.
         */
        msg[1] = omode; /* only root's matters */

        TRACE_COMM(MPI_Bcast)(&msg, 2, MPI_INT, 0, comm);
        NCMPII_HANDLE_ERROR("MPI_Bcast")

        /* check format error (a fatal error, must return now) */
        format = msg[0];
        if (format < 0) return format; /* all netCDF errors are negative */

        /* check omode consistency */
        root_omode = msg[1];
        if (root_omode != omode) {
            omode = root_omode;
            DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_OMODE)
        }

        if (safe_mode) { /* sync status among all processes */
            err = status;
            TRACE_COMM(MPI_Allreduce)(&err, &status, 1, MPI_INT, MPI_MIN, comm);
            NCMPII_HANDLE_ERROR("MPI_Allreduce")
        }
        /* continue to use root's omode to open the file, but will report omode
         * inconsistency error, if there is any */
    }

    /* combine user's MPI info and PNETCDF_HINTS env variable */
    combine_env_hints(info, &combined_info);

#ifdef BUILD_DRIVER_FOO
    if (combined_info == MPI_INFO_NULL)
        MPI_Info_create(&combined_info);
    {
        char value[MPI_MAX_INFO_VAL];
        int flag;

        /* check if nc_foo_driver is enabled */
        MPI_Info_get(combined_info, "nc_foo_driver", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag && strcasecmp(value, "enable") == 0)
            enable_foo_driver = 1;

    }
#endif
#ifdef ENABLE_BURST_BUFFER
    if (combined_info == MPI_INFO_NULL)
        MPI_Info_create(&combined_info);
    {
        char value[MPI_MAX_INFO_VAL];
        int flag;

        /* check if nc_burst_buf is enabled */
        MPI_Info_get(combined_info, "nc_burst_buf", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag && strcasecmp(value, "enable") == 0)
            enable_bb_driver = 1;
    }
#endif

#ifdef ENABLE_NETCDF4
    if (format == NC_FORMAT_NETCDF4_CLASSIC || format == NC_FORMAT_NETCDF4) {
        driver = nc4io_inq_driver();
#ifdef ENABLE_BURST_BUFFER
        /* Burst buffering does not support NetCDF-4 files yet.
         * If hint nc_burst_buf is enabled in combined_info, disable it.
         */
        if (enable_bb_driver == 1)
            MPI_Info_set(combined_info, "nc_burst_buf", "disable");
        enable_bb_driver = 0;
#endif
    }
    else
#else
    if (format == NC_FORMAT_NETCDF4_CLASSIC || format == NC_FORMAT_NETCDF4)
        DEBUG_RETURN_ERROR(NC_ENOTBUILT)
    else
#endif
#ifdef BUILD_DRIVER_FOO
    if (enable_foo_driver)
        driver = ncfoo_inq_driver();
    else
#endif
#ifdef ENABLE_BURST_BUFFER
    if (enable_bb_driver)
        driver = ncbbio_inq_driver();
    else
#endif
    {
        /* ncmpio driver */
        if (format == NC_FORMAT_CLASSIC ||
            format == NC_FORMAT_CDF2 ||
            format == NC_FORMAT_CDF5) {
            driver = ncmpio_inq_driver();
        }
#ifdef ENABLE_ADIOS
        else if (format == NC_FORMAT_BP) {
            driver = ncadios_inq_driver();
        }
#endif
        else /* unrecognized file format */
            DEBUG_RETURN_ERROR(NC_ENOTNC)
    }

    /* allocate a PNC object */
    pncp = (PNC*) NCI_Malloc(sizeof(PNC));
    if (pncp == NULL) {
        *ncidp = -1;
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    /* generate a new nc file ID from NCPList */
    err = new_id_PNCList(ncidp, pncp);
    if (err != NC_NOERR) return err;

    /* Duplicate comm, because users may free it (though unlikely). Note
     * MPI_Comm_dup() is collective. We pass pncp->comm to drivers, so there
     * is no need for a driver to duplicate it again.
     */
    if (comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF)
        MPI_Comm_dup(comm, &pncp->comm);
    else
        pncp->comm = comm;

    /* calling the driver's open subroutine */
    err = driver->open(pncp->comm, path, omode, *ncidp, combined_info, &ncp);
    if (status == NC_NOERR) status = err;
    if (combined_info != MPI_INFO_NULL) MPI_Info_free(&combined_info);
    if (status != NC_NOERR && status != NC_EMULTIDEFINE_OMODE &&
        status != NC_ENULLPAD) {
        /* NC_EMULTIDEFINE_OMODE and NC_ENULLPAD are not fatal error. We
         * continue the rest open procedure */
        del_from_PNCList(*ncidp);
        if (pncp->comm != MPI_COMM_WORLD && pncp->comm != MPI_COMM_SELF)
            MPI_Comm_free(&pncp->comm); /* a collective call */
        NCI_Free(pncp);
        *ncidp = -1;
        return status;
    }

    /* fill in pncp members */
    pncp->path = (char*) NCI_Malloc(strlen(path)+1);
    if (pncp->path == NULL) {
        driver->close(ncp); /* close file and ignore error */
        del_from_PNCList(*ncidp);
        if (pncp->comm != MPI_COMM_WORLD && pncp->comm != MPI_COMM_SELF)
            MPI_Comm_free(&pncp->comm); /* a collective call */
        NCI_Free(pncp);
        *ncidp = -1;
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }
    strcpy(pncp->path, path);
    pncp->mode       = omode;
    pncp->driver     = driver;
    pncp->ndims      = 0;
    pncp->unlimdimid = -1;
    pncp->nvars      = 0;
    pncp->nrec_vars  = 0;
    pncp->vars       = NULL;
    pncp->flag       = 0;
    pncp->ncp        = ncp;
    pncp->format     = format;

    if (!fIsSet(omode, NC_WRITE)) pncp->flag |= NC_MODE_RDONLY;
    if (safe_mode)                pncp->flag |= NC_MODE_SAFE;
    if (!relax_coord_bound)       pncp->flag |= NC_MODE_STRICT_COORD_BOUND;

    /* inquire number of dimensions, variables defined and rec dim ID */
    err = driver->inq(pncp->ncp, &pncp->ndims, &pncp->nvars, NULL,
                      &pncp->unlimdimid);
    if (err != NC_NOERR) goto fn_exit;

    if (pncp->nvars == 0) return status; /* no variable defined in the file */

    /* make a copy of variable metadata at the dispatcher layer, because
     * sanity check is done at the dispatcher layer
     */

    /* allocate chunk size for pncp->vars[] */
    nalloc = _RNDUP(pncp->nvars, PNC_VARS_CHUNK);
    pncp->vars = NCI_Malloc(nalloc * sizeof(PNC_var));
    if (pncp->vars == NULL) {
        DEBUG_ASSIGN_ERROR(err, NC_ENOMEM)
        goto fn_exit;
    }

    dimids = DIMIDS;

    /* construct array of PNC_var for all variables */
    for (i=0; i<pncp->nvars; i++) {
        int ndims, max_ndims=_NDIMS_;
        pncp->vars[i].shape  = NULL;
        pncp->vars[i].recdim = -1;   /* if fixed-size variable */
        err = driver->inq_var(pncp->ncp, i, NULL, &pncp->vars[i].xtype, &ndims,
                              NULL, NULL, NULL, NULL, NULL);
        if (err != NC_NOERR) break; /* loop i */
        pncp->vars[i].ndims = ndims;

        if (ndims > 0) {
            pncp->vars[i].shape = (MPI_Offset*)
                                  NCI_Malloc(ndims * SIZEOF_MPI_OFFSET);
            if (ndims > max_ndims) { /* avoid repeated malloc */
                if (dimids == DIMIDS) dimids = NULL;
                dimids = (int*) NCI_Realloc(dimids, ndims * SIZEOF_INT);
                max_ndims = ndims;
            }
            err = driver->inq_var(pncp->ncp, i, NULL, NULL, NULL,
                                  dimids, NULL, NULL, NULL, NULL);
            if (err != NC_NOERR) break; /* loop i */
            if (dimids[0] == pncp->unlimdimid)
                pncp->vars[i].recdim = pncp->unlimdimid;
            for (j=0; j<ndims; j++) {
                /* obtain size of dimension j */
                err = driver->inq_dim(pncp->ncp, dimids[j], NULL,
                                      pncp->vars[i].shape+j);
                if (err != NC_NOERR) break; /* loop i */
            }
        }
        if (pncp->vars[i].recdim >= 0) pncp->nrec_vars++;
    }
    if (err != NC_NOERR) { /* error happens in loop i */
        assert(i < pncp->nvars);
        for (j=0; j<=i; j++) {
            if (pncp->vars[j].shape != NULL)
                NCI_Free(pncp->vars[j].shape);
        }
        NCI_Free(pncp->vars);
    }
    if (dimids != DIMIDS) NCI_Free(dimids);

fn_exit:
    if (err != NC_NOERR) {
        driver->close(ncp); /* close file and ignore error */
        if (pncp->comm != MPI_COMM_WORLD && pncp->comm != MPI_COMM_SELF)
            MPI_Comm_free(&pncp->comm); /* a collective call */
        del_from_PNCList(*ncidp);
        NCI_Free(pncp->path);
        NCI_Free(pncp);
        *ncidp = -1;
        if (status == NC_NOERR) status = err;
    }

    return status;
}

/*----< ncmpi_close() >------------------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi_close(int ncid)
{
    int i, err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_close() */

    err = pncp->driver->close(pncp->ncp);

    /* Remove from the PNCList, even if err != NC_NOERR */
    del_from_PNCList(ncid);

    /* free the PNC object */
    if (pncp->comm != MPI_COMM_WORLD && pncp->comm != MPI_COMM_SELF)
        MPI_Comm_free(&pncp->comm); /* a collective call */

    NCI_Free(pncp->path);
    for (i=0; i<pncp->nvars; i++)
        if (pncp->vars[i].shape != NULL)
            NCI_Free(pncp->vars[i].shape);
    if (pncp->vars != NULL)
        NCI_Free(pncp->vars);
    NCI_Free(pncp);
  
    return err;
}

/*----< ncmpi_enddef() >-----------------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi_enddef(int ncid) {
    int err=NC_NOERR;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (!(pncp->flag & NC_MODE_DEF)) DEBUG_ASSIGN_ERROR(err, NC_ENOTINDEFINE)

    if (pncp->flag & NC_MODE_SAFE) { /* safe mode */
        int minE, mpireturn;
        /* check the error code across processes */
        TRACE_COMM(MPI_Allreduce)(&err, &minE, 1, MPI_INT, MPI_MIN, pncp->comm);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_error_mpi2nc(mpireturn, "MPI_Allreduce");
        if (minE != NC_NOERR) return minE;
    }
    else if (err != NC_NOERR) return err; /* fatal error */

    /* ---------------------------------------------- META: serilize local metadata to buffer----------------------------------------------*/
    struct hdr local_hdr;

    err = baseline_extract_meta(pncp->ncp, &local_hdr);
    // printf("%s\n", local_hdr.dims.value[0]->name);
    int rank, size;
    MPI_Comm_rank(pncp->comm, &rank);
    MPI_Comm_size(pncp->comm, &size);
    // if (rank > 1){
    // for (int i = 0; i < local_hdr.dims.ndefined; i++) {
    //     printf("rank %d:  Name: %s, Size: %lld\n", rank,  local_hdr.dims.value[i]->name, local_hdr.dims.value[i]->size);
    // }

    // printf("    Variales:\n");
    // for (int i = 0; i < local_hdr.vars.ndefined; i++) {
    //     printf("rank %d;  Name: %s, Type: %d, NumDims: %d\n", rank, local_hdr.vars.value[i]->name,  local_hdr.vars.value[i]->xtype, 
    //     local_hdr.vars.value[i]->ndims);
    //     printf("    Dim IDs: ");
    //     for (int j = 0; j < local_hdr.vars.value[i]->ndims; j++) {
    //         printf("%d ", local_hdr.vars.value[i]->dimids[j]);
    //     }
    //     printf("\n");
    //     printf("    Attributes:\n");
    //     for (int k = 0; k < local_hdr.vars.value[i]->attrs.ndefined; k++) {
    //         printf("      Name: %s, Nelems: %lld, Type: %d\n", local_hdr.vars.value[i]->attrs.value[k]->name, 
    //         local_hdr.vars.value[i]->attrs.value[k]->nelems, local_hdr.vars.value[i]->attrs.value[k]->xtype);
    //     }
    // }
    // }
    char* send_buffer = (char*) NCI_Malloc(local_hdr.xsz);
    err = serialize_hdr(&local_hdr, send_buffer);
    /* ---------------------------------------------- META: Communicate metadata size----------------------------------------------*/

  // Phase 1: Communicate the sizes of the header structure for each process
    MPI_Offset* all_collection_sizes = (MPI_Offset*) NCI_Malloc(size * sizeof(MPI_Offset));
    int mpireturn;
    TRACE_COMM(MPI_Allgather)(&local_hdr.xsz, 1, MPI_OFFSET, all_collection_sizes, 1, MPI_OFFSET, pncp->comm);
    
    /* ---------------------------------------------- META: Communicate metadata ----------------------------------------------*/
    // Calculate displacements for the second phase
    int* recv_displs = (int*) NCI_Malloc(size * sizeof(int));
    int total_recv_size = all_collection_sizes[0];
    recv_displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        recv_displs[i] = recv_displs[i - 1] + all_collection_sizes[i - 1];
        total_recv_size += all_collection_sizes[i];
        
    }
    char* all_collections_buffer = (char*) NCI_Malloc(total_recv_size);

    int* recvcounts =  (int*)NCI_Malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (int)all_collection_sizes[i];
    }
    // Phase 2: Communicate the actual header data
    // Before MPI_Allgatherv
    TRACE_COMM(MPI_Allgatherv)(send_buffer, local_hdr.xsz, MPI_BYTE, all_collections_buffer, recvcounts, recv_displs, MPI_BYTE, pncp->comm);

  /* ---------------------------------------------- META: Deseralize metadata ----------------------------------------------*/

    if (err != NC_NOERR) return err;
        /* allocate buffer for header object NC */
    // NC_dimarray *ncdims = (NC_dimarray*) NCI_Calloc(1, sizeof(NC_dimarray));
    // NC_vararray *ncvars = (NC_vararray*) NCI_Calloc(1, sizeof(NC_vararray));
    // ncdims->ndefined = 0;
    // ncdims->unlimited_id = -1;
    // ncvars->ndefined = 0;
    //Duplicate old header dim array here
    NC *ncp=(NC*)pncp->ncp;
    NC_dimarray *old_dimarray = NCI_Malloc(sizeof(NC_dimarray));
    NC_vararray *old_vararray = NCI_Malloc(sizeof(NC_vararray));

    err = ncmpio_dup_NC_dimarray(old_dimarray, &ncp->dims);
    if (err != NC_NOERR) return err;
    
    err = ncmpio_dup_NC_vararray(old_vararray, &ncp->vars);
    if (err != NC_NOERR) return err;
    
    ncmpio_free_NC_dimarray(&ncp->dims);

    ncmpio_free_NC_vararray(&ncp->vars);
    pncp->ndims      = 0;
    pncp->unlimdimid = -1;
    pncp->nvars      = 0;
    pncp->nrec_vars  = 0;
    pncp->vars       = NULL;
    // pncp->ncp.dims = *ncdims;
    // pncp->ncp->vars = *ncvars;

    for (int i = 0; i < size; ++i) {
        struct hdr recv_hdr;
        // printf("rank %d, recv_displs: %d, recvcounts: %d \n",  rank, recv_displs[i], recvcounts[i]);
        deserialize_hdr(&recv_hdr, all_collections_buffer + recv_displs[i], recvcounts[i]);
        err = add_hdr(&recv_hdr, i, rank, pncp, old_dimarray, old_vararray);
        if (err != NC_NOERR) return err;
    }

    // #ifndef SEARCH_NAME_LINEARLY
    //     /* initialize and populate name lookup tables ---------------------------*/
        // ncmpio_hash_table_populate_NC_dim(&ncp->dims);
        // ncmpio_hash_table_populate_NC_var(&ncp->vars);

    // #endif
    //update local id to index mapping based on index to local id mapping

    //for (int j = 0; j < ncp->dims.ndefined; j++) ncp->dims.indexes[ncp->dims.localids[j]] = j;
    
    // for (int j = 0; j < ncp->vars.ndefined; j++) printf("\n rank: %d:  %d : %d", rank, j, ncp->vars.localids[j]);
    for (int j = 0; j < ncp->dims.ndefined; j++) printf("\n rank: %d:  %d : %d", rank, j, ncp->dims.localids[j]);
    for (int j = 0; j < ncp->dims.ndefined; j++) ncp->dims.indexes[ncp->dims.localids[j]] = j;

    for (int j = 0; j < ncp->vars.ndefined; j++) ncp->vars.indexes[ncp->vars.localids[j]] = j;

    ncmpio_free_NC_dimarray(old_dimarray);
    ncmpio_free_NC_vararray(old_vararray);


    NCI_Free(all_collections_buffer);
    NCI_Free(send_buffer);
    NCI_Free(all_collection_sizes);

    
    

    
    


    /* calling the subroutine that implements ncmpi_enddef() */
    err = pncp->driver->enddef(pncp->ncp);

    if (err != NC_NOERR) return err;

    fClr(pncp->flag, NC_MODE_INDEP); /* default enters collective data mode */
    fClr(pncp->flag, NC_MODE_DEF);

    return NC_NOERR;
}

/*----< ncmpi__enddef() >----------------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi__enddef(int        ncid,
              MPI_Offset h_minfree,
              MPI_Offset v_align,
              MPI_Offset v_minfree,
              MPI_Offset r_align)
{
    int err=NC_NOERR;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (!(pncp->flag & NC_MODE_DEF)) {
        DEBUG_ASSIGN_ERROR(err, NC_ENOTINDEFINE)
        goto err_check;
    }

    if (h_minfree < 0 || v_align < 0 || v_minfree < 0 || r_align < 0) {
        DEBUG_ASSIGN_ERROR(err, NC_EINVAL)
        goto err_check;
    }

err_check:
    if (pncp->flag & NC_MODE_SAFE) { /* safe mode */
        int minE, mpireturn;
        MPI_Offset root_args[4];

        /* first check the error code across processes */
        TRACE_COMM(MPI_Allreduce)(&err, &minE, 1, MPI_INT, MPI_MIN, pncp->comm);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_error_mpi2nc(mpireturn, "MPI_Allreduce");
        if (minE != NC_NOERR) return minE;

        /* check if h_minfree, v_align, v_minfree, and r_align are consistent
         * among all processes */
        root_args[0] = h_minfree;
        root_args[1] = v_align;
        root_args[2] = v_minfree;
        root_args[3] = r_align;
        TRACE_COMM(MPI_Bcast)(&root_args, 4, MPI_OFFSET, 0, pncp->comm);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_error_mpi2nc(mpireturn, "MPI_Bcast");

        if (root_args[0] != h_minfree ||
            root_args[1] != v_align   ||
            root_args[2] != v_minfree ||
            root_args[3] != r_align)
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_FNC_ARGS)

        /* find min error code across processes */
        TRACE_COMM(MPI_Allreduce)(&err, &minE, 1, MPI_INT, MPI_MIN, pncp->comm);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_error_mpi2nc(mpireturn, "MPI_Allreduce");
        if (minE != NC_NOERR) return minE;
    }
    else if (err != NC_NOERR) return err; /* fatal error */

    /* calling the subroutine that implements ncmpi__enddef() */
    err = pncp->driver->_enddef(pncp->ncp, h_minfree, v_align,
                                           v_minfree, r_align);
    if (err != NC_NOERR) return err;

    fClr(pncp->flag, NC_MODE_INDEP); /* default enters collective data mode */
    fClr(pncp->flag, NC_MODE_DEF);
    return NC_NOERR;
}

/*----< ncmpi_redef() >------------------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi_redef(int ncid)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (fIsSet(pncp->flag, NC_MODE_RDONLY)) /* read-only */
        DEBUG_RETURN_ERROR(NC_EPERM)
    /* if open mode is inconsistent, then this return might cause parallel
     * program to hang */

    /* cannot be in define mode, must enter from data mode */
    if (fIsSet(pncp->flag, NC_MODE_DEF)) DEBUG_RETURN_ERROR(NC_EINDEFINE)

    /* calling the subroutine that implements ncmpi_redef() */
    err = pncp->driver->redef(pncp->ncp);
    if (err != NC_NOERR) return err;

    fSet(pncp->flag, NC_MODE_DEF);
    return NC_NOERR;
}

/*----< ncmpi_sync() >-------------------------------------------------------*/
/* This API is a collective subroutine, and must be called in data mode, no
 * matter if it is in collective or independent data mode.
 */
int
ncmpi_sync(int ncid)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_sync() */
    return pncp->driver->sync(pncp->ncp);
}

/*----< ncmpi_flush() >-------------------------------------------------------*/
/* This API is a collective subroutine, and must be called in data mode, no
 * matter if it is in collective or independent data mode.
 */
int
ncmpi_flush(int ncid)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_flush() */
    return pncp->driver->flush(pncp->ncp);
}

/*----< ncmpi_abort() >------------------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi_abort(int ncid)
{
    int i, err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_abort() */
    err = pncp->driver->abort(pncp->ncp);

    /* Remove from the PNCList, even if err != NC_NOERR */
    del_from_PNCList(ncid);

    /* free the PNC object */
    if (pncp->comm != MPI_COMM_WORLD && pncp->comm != MPI_COMM_SELF)
        MPI_Comm_free(&pncp->comm); /* a collective call */

    NCI_Free(pncp->path);
    for (i=0; i<pncp->nvars; i++)
        if (pncp->vars[i].shape != NULL)
            NCI_Free(pncp->vars[i].shape);
    if (pncp->vars != NULL)
        NCI_Free(pncp->vars);
    NCI_Free(pncp);

    return err;
}

/*----< ncmpi_set_fill() >---------------------------------------------------*/
/* This is a collective subroutine.
 * This subroutine serves both purposes of setting and inquiring the fill mode.
 */
int
ncmpi_set_fill(int  ncid,
               int  fill_mode,     /* mode to be changed by user */
               int *old_fill_mode) /* current fill mode */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (fIsSet(pncp->flag, NC_MODE_RDONLY)) /* read-only */
        DEBUG_RETURN_ERROR(NC_EPERM)

    /* not allowed to call in data mode for classic formats */
    if ((pncp->format != NC_FORMAT_NETCDF4) && !(pncp->flag & NC_MODE_DEF))
        DEBUG_RETURN_ERROR(NC_ENOTINDEFINE)

    /* calling the subroutine that implements ncmpi_set_fill() */
    err = pncp->driver->set_fill(pncp->ncp, fill_mode, old_fill_mode);
    if (err != NC_NOERR) return err;

    if (fill_mode == NC_FILL)
        fSet(pncp->flag, NC_MODE_FILL);
    else /* NC_NOFILL */
        fClr(pncp->flag, NC_MODE_FILL);

    return NC_NOERR;
}

/*----< ncmpi_inq_format() >-------------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_format(int  ncid,
                 int *formatp)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (formatp != NULL) *formatp = pncp->format;

    return NC_NOERR;
}

#ifdef ENABLE_ADIOS
static void swap_64(void *data)
{
    uint64_t *dest = (uint64_t*) data;
    uint64_t tmp;
    memcpy(&tmp, dest, 8);
    *dest = ((tmp & 0x00000000000000FFULL) << 56) |
            ((tmp & 0x000000000000FF00ULL) << 40) |
            ((tmp & 0x0000000000FF0000ULL) << 24) |
            ((tmp & 0x00000000FF000000ULL) <<  8) |
            ((tmp & 0x000000FF00000000ULL) >>  8) |
            ((tmp & 0x0000FF0000000000ULL) >> 24) |
            ((tmp & 0x00FF000000000000ULL) >> 40) |
            ((tmp & 0xFF00000000000000ULL) >> 56);
}

static int adios_parse_endian(char *footer, int *diff_endianness) {
    unsigned int version;
    unsigned int test = 1; /* If high bit set, big endian */

    version = ntohl (*(uint32_t *) (footer + BP_MINIFOOTER_SIZE - 4));
    char *v = (char *) (&version);
    if ((*v && !*(char *) &test) /* Both writer and reader are big endian */
        || (!*(v+3) && *(char *) &test)){ /* Both are little endian */
        *diff_endianness = 0; /* No need to change endianness */
    }
    else{
        *diff_endianness = 1;
    }

    return 0;
}
#endif

/*----< ncmpi_inq_file_format() >--------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_file_format(const char *filename,
                      int        *formatp) /* out */
{
    const char *cdf_signature="CDF";
    const char *hdf5_signature="\211HDF\r\n\032\n";
    const char *path;
    char signature[8];
    int fd;
    ssize_t rlen;

    if (formatp == NULL) return NC_NOERR;

    *formatp = NC_FORMAT_UNKNOWN;

    /* remove the file system type prefix name if there is any.  For example,
     * when filename = "lustre:/home/foo/testfile.nc", remove "lustre:" to make
     * path pointing to "/home/foo/testfile.nc", so it can be used in POSIX
     * open() below
     */
    path = ncmpii_remove_file_system_type_prefix(filename);

    /* must include config.h on 32-bit machines, as AC_SYS_LARGEFILE is called
     * at the configure time and it defines _FILE_OFFSET_BITS to 64 if large
     * file feature is supported.
     */
    if ((fd = open(path, O_RDONLY, 00400)) == -1) { /* open for read */
             if (errno == ENOENT)       DEBUG_RETURN_ERROR(NC_ENOENT)
        else if (errno == EACCES)       DEBUG_RETURN_ERROR(NC_EACCESS)
        else if (errno == ENAMETOOLONG) DEBUG_RETURN_ERROR(NC_EBAD_FILE)
        else {
            fprintf(stderr,"Error on opening file %s (%s)\n",
                    filename,strerror(errno));
            DEBUG_RETURN_ERROR(NC_EFILE)
        }
    }
    /* get first 8 bytes of file */
    rlen = read(fd, signature, 8);
    if (rlen != 8) {
        close(fd); /* ignore error */
        DEBUG_RETURN_ERROR(NC_EFILE)
    }
    if (close(fd) == -1) {
        DEBUG_RETURN_ERROR(NC_EFILE)
    }

    if (memcmp(signature, cdf_signature, 3) == 0) {
             if (signature[3] == 5)  *formatp = NC_FORMAT_CDF5;
        else if (signature[3] == 2)  *formatp = NC_FORMAT_CDF2;
        else if (signature[3] == 1)  *formatp = NC_FORMAT_CLASSIC;
    }

    /* check if the file is an HDF5. */
    if (*formatp == NC_FORMAT_UNKNOWN) {
        /* The HDF5 superblock is located by searching for the HDF5 format
         * signature at byte offset 0, byte offset 512, and at successive
         * locations in the file, each a multiple of two of the previous
         * location; in other words, at these byte offsets: 0, 512, 1024, 2048,
         * and so on. The space before the HDF5 superblock is referred as to
         * "user block".
         */
        off_t offset=0;

        fd = open(path, O_RDONLY, 00400); /* error check already done */
        /* get first 8 bytes of file */
        rlen = read(fd, signature, 8); /* error check already done */

        while (rlen == 8 && memcmp(signature, hdf5_signature, 8)) {
            offset = (offset == 0) ? 512 : offset * 2;
            lseek(fd, offset, SEEK_SET);
            rlen = read(fd, signature, 8);
        }
        close(fd); /* ignore error */

        if (rlen == 8) { /* HDF5 signature found */
            /* TODO: whether the file is NC_FORMAT_NETCDF4_CLASSIC is
             * determined by HDF5 attribute "_nc3_strict" which requires a call
             * to H5Aget_name(). For now, we do not distinguish
             * NC_CLASSIC_MODEL, but simply return NETCDF4 format.
             */
#ifdef ENABLE_NETCDF4
            int err, ncid;
            err = nc_open(path, NC_NOWRITE, &ncid);
            if (err != NC_NOERR) DEBUG_RETURN_ERROR(err)
            err = nc_inq_format(ncid, formatp);
            if (err != NC_NOERR) DEBUG_RETURN_ERROR(err)
            err = nc_close(ncid);
            if (err != NC_NOERR) DEBUG_RETURN_ERROR(err)
#else
            *formatp = NC_FORMAT_NETCDF4;
#endif
        }
    }

#ifdef ENABLE_ADIOS
    /* check if the file is a BP. */
    if (*formatp == NC_FORMAT_UNKNOWN) {
        off_t fsize;
        int diff_endian;
        char footer[BP_MINIFOOTER_SIZE];
        off_t h1, h2, h3;

        /* test if the file footer follows BP specification */
        if ((fd = open(path, O_RDONLY, 00400)) == -1) {
                 if (errno == ENOENT)       DEBUG_RETURN_ERROR(NC_ENOENT)
            else if (errno == EACCES)       DEBUG_RETURN_ERROR(NC_EACCESS)
            else if (errno == ENAMETOOLONG) DEBUG_RETURN_ERROR(NC_EBAD_FILE)
            else {
                fprintf(stderr,"Error on opening file %s (%s)\n",
                        filename,strerror(errno));
                DEBUG_RETURN_ERROR(NC_EFILE)
            }
        }

        /* Seek to end of file */
        fsize = lseek(fd, (off_t)(-(BP_MINIFOOTER_SIZE)), SEEK_END);

        /* read footer */
        rlen = read(fd, footer, BP_MINIFOOTER_SIZE);
        if (rlen != BP_MINIFOOTER_SIZE) {
            close(fd);
            DEBUG_RETURN_ERROR(NC_EFILE)
        }
        if (close(fd) == -1) {
            DEBUG_RETURN_ERROR(NC_EFILE)
        }

        /* check endianness of file and this running system */
        adios_parse_endian(footer, &diff_endian);

        BUFREAD64(footer,      h1) /* file offset of process group index table */
        BUFREAD64(footer + 8,  h2) /* file offset of variable index table */
        BUFREAD64(footer + 16, h3) /* file offset of attribute index table */

        /* All index tables must fall within the range of file size.
         * Process group index table must comes before variable index table.
         * Variable index table must comes before attribute index table.
         */
        if (0 < h1 && h1 < fsize &&
            0 < h2 && h2 < fsize &&
            0 < h3 && h3 < fsize &&
            h1 < h2 && h2 < h3){
            /* basic footer check is passed, now we try to open the file with
             * ADIOS library to make sure it is indeed a BP formated file
             */
            ADIOS_FILE *fp;
            fp = adios_read_open_file(path, ADIOS_READ_METHOD_BP,
                                        MPI_COMM_SELF);
            if (fp != NULL) {
                *formatp = NC_FORMAT_BP;
                adios_read_close(fp);
            }
        }
    }
#endif

    return NC_NOERR;
}

/*----< ncmpi_inq_version() >------------------------------------------------*/
int
ncmpi_inq_version(int ncid, int *nc_mode)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (nc_mode == NULL) return NC_NOERR;

    if (pncp->format == NC_FORMAT_CDF5)
        *nc_mode = NC_64BIT_DATA;
    else if (pncp->format == NC_FORMAT_CDF2)
        *nc_mode = NC_64BIT_OFFSET;
    else if (pncp->format == NC_FORMAT_CLASSIC)
        *nc_mode = NC_CLASSIC_MODEL;

#ifdef ENABLE_NETCDF4
    else if (pncp->format == NC_FORMAT_NETCDF4)
        *nc_mode = NC_NETCDF4;
    else if (pncp->format == NC_FORMAT_NETCDF4_CLASSIC)
        *nc_mode = NC_NETCDF4 | NC_CLASSIC_MODEL;
#endif

#ifdef ENABLE_ADIOS
    else if (pncp->format == NC_FORMAT_BP)
        *nc_mode = NC_BP;
#endif

    return NC_NOERR;
}

/*----< ncmpi_inq() >--------------------------------------------------------*/
int
ncmpi_inq(int  ncid,
          int *ndimsp,
          int *nvarsp,
          int *nattsp,
          int *xtendimp)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_inq() */
    return pncp->driver->inq(pncp->ncp, ndimsp, nvarsp, nattsp, xtendimp);
}

/*----< ncmpi_inq_ndims() >--------------------------------------------------*/
int
ncmpi_inq_ndims(int  ncid,
                int *ndimsp)
{
    return ncmpi_inq(ncid, ndimsp, NULL, NULL, NULL);
}

/*----< ncmpi_inq_nvars() >--------------------------------------------------*/
int
ncmpi_inq_nvars(int  ncid,
                int *nvarsp)
{
    return ncmpi_inq(ncid, NULL, nvarsp, NULL, NULL);
}

/*----< ncmpi_inq_natts() >--------------------------------------------------*/
int
ncmpi_inq_natts(int  ncid,
                int *nattsp)
{
    return ncmpi_inq(ncid, NULL, NULL, nattsp, NULL);
}

/*----< ncmpi_inq_unlimdim() >-----------------------------------------------*/
int
ncmpi_inq_unlimdim(int  ncid,
                   int *unlimdimidp)
{
    return ncmpi_inq(ncid, NULL, NULL, NULL, unlimdimidp);
}

/*----< ncmpi_inq_path() >---------------------------------------------------*/
/* Get the file pathname which was used to open/create the ncid's file.
 * pathlen and path must already be allocated. Ignored if NULL.
 * This is an independent subroutine.
 */
int
ncmpi_inq_path(int   ncid,
               int  *pathlen,/* Ignored if NULL */
               char *path)   /* must have already been allocated. Ignored if NULL */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

#if 0
    /* calling the subroutine that implements ncmpi_inq_path() */
    return pncp->driver->inq_misc(pncp->ncp, pathlen, path, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL);
#endif
    if (pathlen != NULL) {
        if (pncp->path == NULL) *pathlen = 0;
        else                    *pathlen = (int)strlen(pncp->path);
    }
    if (path != NULL) {
        if (pncp->path == NULL) *path = '\0';
        else                    strcpy(path, pncp->path);
    }
    return NC_NOERR;
}

/*----< ncmpi_inq_num_fix_vars() >-------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_num_fix_vars(int ncid, int *num_fix_varsp)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (num_fix_varsp == NULL) return NC_NOERR;

#ifdef ENABLE_NETCDF4
    if (pncp->format == NC_FORMAT_NETCDF4 ||
        pncp->format == NC_FORMAT_NETCDF4_CLASSIC) {
        /* calling the subroutine that implements ncmpi_inq_num_fix_vars() */
        return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, num_fix_varsp,
                                      NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                      NULL, NULL, NULL, NULL, NULL);
    }
#endif

    *num_fix_varsp = pncp->nvars - pncp->nrec_vars;

    /* number of fixed-size variables can also be calculated below.
    int i;
    *num_fix_varsp = 0;
    for (i=0; i<pncp->nvars; i++) {
        if (pncp->vars[i].recdim < 0)
            (*num_fix_varsp)++;
    }
    */

    return NC_NOERR;
}

/*----< ncmpi_inq_num_rec_vars() >-------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_num_rec_vars(int ncid, int *num_rec_varsp)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (num_rec_varsp == NULL) return NC_NOERR;

#ifdef ENABLE_NETCDF4
    if (pncp->format == NC_FORMAT_NETCDF4 ||
        pncp->format == NC_FORMAT_NETCDF4_CLASSIC) {
        /* calling the subroutine that implements ncmpi_inq_num_rec_vars() */
        return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL,
                                      num_rec_varsp, NULL, NULL, NULL, NULL,
                                      NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        }
#endif

    *num_rec_varsp = pncp->nrec_vars;

    /* number of record variables can also be calculated below.
    int i;
    *num_rec_varsp = 0;
    for (i=0; i<pncp->nvars; i++) {
        if (pncp->vars[i].recdim >= 0)
            (*num_rec_varsp)++;
    }
    */

    return NC_NOERR;
}

/*----< ncmpi_inq_striping() >-----------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_striping(int ncid, int *striping_size, int *striping_count)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_inq_striping() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  striping_size, striping_count, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

/*----< ncmpi_inq_header_size() >--------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_header_size(int ncid, MPI_Offset *header_size)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (header_size == NULL) return NC_NOERR;

    /* calling the subroutine that implements ncmpi_inq_header_size() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  NULL, NULL, header_size, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL);
}

/*----< ncmpi_inq_header_extent() >------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_header_extent(int ncid, MPI_Offset *header_extent)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (header_extent == NULL) return NC_NOERR;

    /* calling the subroutine that implements ncmpi_inq_header_extent() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, header_extent, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL);
}

/*----< ncmpi_inq_recsize() >------------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_recsize(int ncid, MPI_Offset *recsize)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (recsize == NULL) return NC_NOERR;

    /* calling the subroutine that implements ncmpi_inq_recsize() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, recsize, NULL,
                                  NULL, NULL, NULL, NULL, NULL);
}

/*----< ncmpi_inq_put_size() >-----------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_put_size(int ncid, MPI_Offset *put_size)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (put_size == NULL) return NC_NOERR;

    /* calling the subroutine that implements ncmpi_inq_put_size() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL, put_size,
                                  NULL, NULL, NULL, NULL, NULL);
}

/*----< ncmpi_inq_get_size() >-----------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_get_size(int ncid, MPI_Offset *get_size)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (get_size == NULL) return NC_NOERR;

    /* calling the subroutine that implements ncmpi_inq_get_size() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL, NULL,
                                  get_size, NULL, NULL, NULL, NULL);
}

/*----< ncmpi_inq_file_info() >----------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_file_info(int ncid, MPI_Info *info)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (info == NULL) return NC_NOERR;

    /* calling the subroutine that implements ncmpi_inq_file_info() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL, NULL,
                                  NULL, info, NULL, NULL, NULL);
}

/* ncmpi_get_file_info() is now deprecated, replaced by ncmpi_inq_file_info() */
int
ncmpi_get_file_info(int ncid, MPI_Info *info)
{
    return ncmpi_inq_file_info(ncid, info);
}

/*----< ncmpi_begin_indep_data() >-------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi_begin_indep_data(int ncid)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_begin_indep_data() */
    err = pncp->driver->begin_indep_data(pncp->ncp);
    if (err != NC_NOERR) return err;

    fSet(pncp->flag, NC_MODE_INDEP);
    return NC_NOERR;
}

/*----< ncmpi_end_indep_data() >---------------------------------------------*/
/* This is a collective subroutine. */
int
ncmpi_end_indep_data(int ncid)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_end_indep_data() */
    err = pncp->driver->end_indep_data(pncp->ncp);
    if (err != NC_NOERR) return err;

    fClr(pncp->flag, NC_MODE_INDEP);
    return NC_NOERR;
}

/*----< ncmpi_sync_numrecs() >-----------------------------------------------*/
/* this API is collective, but can be called in independent data mode.
 * Note numrecs (number of records) is always sync-ed in memory and file in
 * collective data mode.
 */
int
ncmpi_sync_numrecs(int ncid)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_sync_numrecs() */
    return pncp->driver->sync_numrecs(pncp->ncp);
}

/*----< ncmpi_set_default_format() >-----------------------------------------*/
/* This function sets a default create file format.
 * Valid formats are NC_FORMAT_CLASSIC, NC_FORMAT_CDF2, and NC_FORMAT_CDF5
 * This API is NOT collective, as there is no way to check against an MPI
 * communicator. It should be called by all MPI processes that intend to
 * create a file later. Consistency check will have to be done in other APIs.
 */
int
ncmpi_set_default_format(int format, int *old_formatp)
{
    int err=NC_NOERR, perr=0;

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_lock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_lock")
#endif

    /* Return existing format if desired. */
    if (old_formatp != NULL)
        *old_formatp = ncmpi_default_create_format;

    /* Make sure only valid format is set. */
    if (format != NC_FORMAT_CLASSIC &&
        format != NC_FORMAT_CDF2 &&
        format != NC_FORMAT_NETCDF4 &&
        format != NC_FORMAT_NETCDF4_CLASSIC &&
        format != NC_FORMAT_CDF5) {
        DEBUG_ASSIGN_ERROR(err, NC_EINVAL)
    }
    else {
        ncmpi_default_create_format = format;
        err = NC_NOERR;
    }

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_unlock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_unlock")

err_out:
#endif
    return (err != NC_NOERR) ? err : perr;
}

/*----< ncmpi_inq_default_format() >-----------------------------------------*/
/* returns a value suitable for a create flag.  Will return one or more of the
 * following values OR-ed together: NC_64BIT_OFFSET, NC_CLOBBER */
int
ncmpi_inq_default_format(int *formatp)
{
    int perr=0;

    if (formatp == NULL) DEBUG_RETURN_ERROR(NC_EINVAL)

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_lock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_lock")
#endif

    *formatp = ncmpi_default_create_format;

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_unlock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_unlock")

err_out:
#endif
    return perr;
}

/*----< ncmpi_inq_files_opened() >-------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_files_opened(int *num,    /* cannot be NULL */
                       int *ncids)  /* can be NULL */
{
    int i, perr=0;

    if (num == NULL) DEBUG_RETURN_ERROR(NC_EINVAL)

#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_lock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_lock")
#endif

    *num = pnc_numfiles;

    if (ncids != NULL) { /* ncids can be NULL */
        *num = 0;
        for (i=0; i<NC_MAX_NFILES; i++) {
            if (pnc_filelist[i] != NULL) {
                ncids[*num] = i;
                (*num)++;
            }
        }
    }
#ifdef ENABLE_THREAD_SAFE
    perr = pthread_mutex_unlock(&lock);
    CHECK_ERRNO(perr, "pthread_mutex_unlock")

err_out:
#endif
    return perr;
}

/*----< ncmpi_inq_nreqs() >--------------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_nreqs(int  ncid,
                int *nreqs) /* number of pending nonblocking requests */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (nreqs == NULL) return NC_NOERR;

    /* calling the subroutine that implements ncmpi_inq_nreqs() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL, NULL,
                                  NULL, NULL, nreqs, NULL, NULL);
}

/*----< ncmpi_inq_buffer_usage() >-------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_buffer_usage(int         ncid,
                       MPI_Offset *usage) /* amount of space used so far */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (usage == NULL) return NC_NOERR;

    /* calling the subroutine that implements ncmpi_inq_buffer_usage() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, usage, NULL);
}

/*----< ncmpi_inq_buffer_size() >--------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_buffer_size(int         ncid,
                      MPI_Offset *buf_size) /* amount of space attached */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (buf_size == NULL) return NC_NOERR;

    /* calling the subroutine that implements ncmpi_inq_buffer_size() */
    return pncp->driver->inq_misc(pncp->ncp, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, buf_size);
}

/*----< ncmpi_buffer_attach() >----------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_buffer_attach(int        ncid,
                    MPI_Offset bufsize) /* amount of memory space allowed for
                                           PnetCDF library to buffer the
                                           nonblocking requests */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_buffer_attach() */
    return pncp->driver->buffer_attach(pncp->ncp, bufsize);
}

/*----< ncmpi_buffer_detach() >----------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_buffer_detach(int ncid)
{
    int err;
    PNC *pncp;

    /* check if ncid is valid */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_buffer_detach() */
    return pncp->driver->buffer_detach(pncp->ncp);
}

/*----< ncmpi_delete() >-----------------------------------------------------*/
/*
 * filename: the name of the file we will remove.
 * info: MPI info object, in case underlying file system needs hints.
 *
 * This API is implemented in src/driver/ncmpio/ncmpio_file.c
 *
 */

/*----< ncmpi_wait() >-------------------------------------------------------*/
/* This API is an independent subroutine. */
int
ncmpi_wait(int  ncid,
           int  num_reqs, /* number of requests */
           int *req_ids,  /* [num_reqs]: IN/OUT */
           int *statuses) /* [num_reqs], can be NULL */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid.
     * For invalid ncid, we must return error now, as there is no way to
     * continue with invalid ncp. However, collective APIs might hang if this
     * error occurs only on a subset of processes
     */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_wait() */
    return pncp->driver->wait(pncp->ncp, num_reqs, req_ids, statuses,
                              NC_REQ_INDEP);
}

/*----< ncmpi_wait_all() >---------------------------------------------------*/
/* This API is a collective subroutine. */
int
ncmpi_wait_all(int  ncid,
               int  num_reqs, /* number of requests */
               int *req_ids,  /* [num_reqs]: IN/OUT */
               int *statuses) /* [num_reqs], can be NULL */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid.
     * For invalid ncid, we must return error now, as there is no way to
     * continue with invalid ncp. However, collective APIs might hang if this
     * error occurs only on a subset of processes
     */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_wait_all() */
    return pncp->driver->wait(pncp->ncp, num_reqs, req_ids, statuses,
                              NC_REQ_COLL);
}

/*----< ncmpi_cancel() >-----------------------------------------------------*/
/* This is an independent subroutine */
int
ncmpi_cancel(int  ncid,
             int  num_reqs, /* number of requests */
             int *req_ids,  /* [num_reqs]: IN/OUT */
             int *statuses) /* [num_reqs], can be NULL */
{
    int err;
    PNC *pncp;

    /* check if ncid is valid.
     * For invalid ncid, we must return error now, as there is no way to
     * continue with invalid ncp. However, collective APIs might hang if this
     * error occurs only on a subset of processes
     */
    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    /* calling the subroutine that implements ncmpi_cancel() */
    return pncp->driver->cancel(pncp->ncp, num_reqs, req_ids, statuses);
}

