#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy(), strncpy() */
#include <unistd.h> /* getopt() */
#include <time.h>   /* time() localtime(), asctime() */
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>
#include "baseline_ncx.h" 
#include "../drivers/ncmpio/ncmpio_NC.h"

  /* ---------------------------------- Serializaition ----------------------------------------*/

int
xlen_nc_type(nc_type xtype, int *size)
{
    switch(xtype) {
        case NC_BYTE:
        case NC_CHAR:
        case NC_UBYTE:  *size = 1; return 0;
        case NC_SHORT:
        case NC_USHORT: *size = 2; return 0;
        case NC_INT:
        case NC_UINT:
        case NC_FLOAT:  *size = 4; return 0;
        case NC_DOUBLE:
        case NC_INT64:
        case NC_UINT64: *size = 8; return 0;
    }
    return 0;
}

static int 
putn_text(void **xpp, MPI_Offset nelems, const char *tp)
{
	(void) memcpy(*xpp, tp, (size_t)nelems);
	*xpp = (void *)((char *)(*xpp) + nelems);
	return 0;
}

static int
put_uint32(void **xpp, unsigned int ip)
{
    memcpy(*xpp, &ip, 4);
    /* advace *xpp 4 bytes */
    *xpp  = (void *)((char *)(*xpp) + 4);
    return 0;
}

static int
serialize_dim(metabuffer   *pbp,
               const hdr_dim *dimp)
{
    /* copy name */
    serialize_name(pbp, dimp->name);
    put_uint32((void**)(&pbp->pos), (uint32_t)dimp->size);
    return 0;
}

static int
serialize_name(metabuffer *pbp,
                const char *name)
{
    size_t nchars = strlen(name);

    put_uint32((void**)(&pbp->pos), (uint32_t)nchars);

    return putn_text((void **)(&pbp->pos), (MPI_Offset)nchars, name);
}


static int
serialize_dimarray(metabuffer        *pbp,
                    const hdr_dimarray *ncap)
{
    int i, status;
    assert(pbp != NULL);

    if (ncap == NULL || ncap->ndefined == 0) { /* ABSENT */
        status = put_uint32((void**)(&pbp->pos), NC_UNSPECIFIED);
        if (status != NC_NOERR) return status;
        status = put_uint32((void**)(&pbp->pos), 0);
        if (status != NC_NOERR) return status;
    }
    else {
        /* copy NC_DIMENSION */
        status = put_uint32((void**)(&pbp->pos), NC_DIMENSION);
        if (status != NC_NOERR) return status;
        status = put_uint32((void**)(&pbp->pos), (uint32_t)ncap->ndefined);
        if (status != NC_NOERR) return status;
        /* copy [dim ...] */
        for (i=0; i<ncap->ndefined; i++) {
            status = serialize_dim(pbp, ncap->value[i]);
            if (status != NC_NOERR) return status;
        }
    }
    return 0;
}

static int
serialize_attrV(metabuffer    *pbp,
                 const hdr_attr *attrp)
{

    int xsz;
    int sz;

    /* xlen_nc_type() returns the element size (unaligned) of
     * attrp->xtype attrp->xsz is the aligned total size of attribute values
     */
    xlen_nc_type(attrp->xtype, &xsz);
    sz = attrp->nelems * xsz;
    memcpy(pbp->pos, attrp->xvalue, (size_t)sz);
    pbp->pos = (void *)((char *)pbp->pos + sz);
    return 0;
}

/*----< serialize_NC_attr() >--------------------------------------------------*/
static int
serialize_attr(metabuffer    *pbp,
                const hdr_attr *attrp)
{
    int status;

    /* copy name */
    status = serialize_name(pbp, attrp->name);
    if (status != NC_NOERR) return status;

    /* copy nc_type */
    status = put_uint32((void**)(&pbp->pos), (uint32_t)attrp->xtype);
    if (status != NC_NOERR) return status;

    /* copy nelems */
    status = put_uint32((void**)(&pbp->pos), (uint32_t)attrp->nelems);
    if (status != NC_NOERR) return status;

    /* copy [values ...] */
    status = serialize_attrV(pbp, attrp);
    if (status != NC_NOERR) return status;

    return NC_NOERR;
}

static int
serialize_attrarray(metabuffer         *pbp,
                     const hdr_attrarray *ncap)
{

    int i, status;
    assert(pbp != NULL);
    if (ncap == NULL || ncap->ndefined == 0) { /* ABSENT */
        status = put_uint32((void**)(&pbp->pos), NC_UNSPECIFIED);
        if (status != NC_NOERR) return status;
        status = put_uint32((void**)(&pbp->pos), 0);
        if (status != NC_NOERR) return status;
    }
    else {
        /* copy NC_ATTRIBUTE */
        status = put_uint32((void**)(&pbp->pos), NC_ATTRIBUTE);
        if (status != NC_NOERR) return status;
        status = put_uint32((void**)(&pbp->pos), (uint32_t)ncap->ndefined);
        if (status != NC_NOERR) return status;
        /* copy [attr ...] */
        for (i=0; i<ncap->ndefined; i++) {
            status = serialize_attr(pbp, ncap->value[i]);
            if (status != NC_NOERR) return status;
        }
    }

    return NC_NOERR;
}

static int
serialize_var(metabuffer   *pbp,
               const hdr_var *varp)
{
    int i, status;

    /* copy name */
    status = serialize_name(pbp, varp->name);
    if (status != NC_NOERR) return status;

    /* copy nelems */

    status = put_uint32((void**)(&pbp->pos), (uint32_t)varp->ndims);

    if (status != NC_NOERR) return status;

    /* copy [dim_index ...] i*/
    for (i=0; i<varp->ndims; i++) {

        status = put_uint32((void**)(&pbp->pos), (uint32_t)varp->dimids[i]);
        if (status != NC_NOERR) return status;
    }

    /* copy vatt_list */
    status = serialize_attrarray(pbp, &varp->attrs);
    if (status != NC_NOERR) return status;

    /* copy nc_type */
    status = put_uint32((void**)(&pbp->pos), (uint32_t)varp->xtype);
    if (status != NC_NOERR) return status;

    return NC_NOERR;
}


/*----< serialize_vararray() >----------------------------------------------*/
static int
serialize_vararray(metabuffer        *pbp,
                    const hdr_vararray *ncap)
{
    int i, status;
    assert(pbp != NULL);
    if (ncap == NULL || ncap->ndefined == 0) { /* ABSENT */
        status = put_uint32((void**)(&pbp->pos), NC_UNSPECIFIED);
        if (status != NC_NOERR) return status;
        status = put_uint32((void**)(&pbp->pos), 0);
        if (status != NC_NOERR) return status;
    }
    else {
        /* copy NC_VARIABLE */
        status = put_uint32((void**)(&pbp->pos), NC_VARIABLE);
        if (status != NC_NOERR) return status;

        /* copy nelems */
        status = put_uint32((void**)(&pbp->pos), (uint32_t)ncap->ndefined);
        if (status != NC_NOERR) return status;

        /* copy [var ...] */
        for (i=0; i<ncap->ndefined; i++) {
            status =serialize_var(pbp, ncap->value[i]);
            if (status != NC_NOERR) return status;
        }
    }
    return NC_NOERR;
}



/*----< serialize_hdr() >----------------------------------------------*/
int
serialize_hdr(struct hdr *ncp, void *buf)
{
    int status;
    metabuffer putbuf;

    putbuf.pos           = buf;
    putbuf.base          = buf;
    putbuf.size          = ncp->xsz;

    /* copy dim_list */
    status = serialize_dimarray(&putbuf, &ncp->dims);
    if (status != NC_NOERR) return status;



    // /* copy gatt_list */
    // status = serialize_attrarray(&putbuf, &ncp->attrs);
    // if (status != NC_NOERR) return status;

    /* copy var_list */
    status = serialize_vararray(&putbuf, &ncp->vars);
    if (status != NC_NOERR) return status;

    size_t serializedSize = putbuf.pos - putbuf.base;

    // Print the result
    // printf("Number of bytes taken after serialization: %zu\n", serializedSize);



    return NC_NOERR;
}

  /* ---------------------------------- Deserializaition ----------------------------------------*/


static int
getn_text(void **xpp, MPI_Offset nelems, char *tp)
{
	(void) memcpy(tp, *xpp, (size_t)nelems);
    tp[nelems] = '\0';
	*xpp = (void *)((char *)(*xpp) + nelems);
	return NC_NOERR;

}


static int
get_uint32(void **xpp, unsigned int *ip)
{
    memcpy(ip, *xpp, 4);
    /* advance *xpp 4 bytes */
    *xpp = (void *)((const char *)(*xpp) + 4);
    return NC_NOERR;
}

static int deserialize_nc_type(metabuffer *gbp, nc_type *xtypep){
    int err;
    uint32_t xtype;
    err = get_uint32((void **)(&gbp->pos), &xtype);
    if (err != NC_NOERR) return err;
    *xtypep = (nc_type) xtype;
    return NC_NOERR;
}

static int deserialize_name(metabuffer *gbp, char **name) {
    unsigned int nchars;
    get_uint32((void**)&gbp->pos, &nchars);
    *name = (char *)malloc(nchars + 1);
    if (*name == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    getn_text((void **)&gbp->pos, nchars, *name);
    return NC_NOERR;
}

static int deserialize_dim(metabuffer *gbp, hdr_dim *dimp) {
    MPI_Offset dim_length;
    uint32_t tmp;
    char *name;
    int err;
    err = deserialize_name(gbp, &name); 
    if (err != NC_NOERR) return err;
    get_uint32((void**)&gbp->pos, &tmp);
    dim_length = (MPI_Offset)tmp;
    dimp->name     = name;
    dimp->name_len = strlen(name);
    dimp->size     = dim_length;
    return 0;
}

static int deserialize_dimarray(metabuffer *gbp, hdr_dimarray *ncap) {
    unsigned int tag;
    get_uint32((void**)&gbp->pos, &tag);
    if (tag == NC_UNSPECIFIED) {
        get_uint32((void**)&gbp->pos, (unsigned int *)&ncap->ndefined);
        assert(ncap->ndefined == 0);
        return 0; // ABSENT
    } else if (tag == NC_DIMENSION) {
        get_uint32((void**)&gbp->pos, (unsigned int *)&ncap->ndefined);

        ncap->value = (hdr_dim **)malloc(ncap->ndefined * sizeof(hdr_dim *));
        if (ncap->value == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return -1;
        }

        for (int i = 0; i < ncap->ndefined; i++) {
            ncap->value[i] = (hdr_dim *)malloc(sizeof(hdr_dim));
            if (ncap->value[i] == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                return -1;
            }
            if (deserialize_dim(gbp, ncap->value[i]) != 0) {
                return -1;
            }
        }
    }

    return 0;
}

static int deserialize_attrV(metabuffer *gbp, hdr_attr *attrp) {
    int xsz, sz, err;

    xlen_nc_type(attrp->xtype, &xsz);
    sz = attrp->nelems * xsz;

    attrp->xvalue = malloc(sz);
    if (attrp->xvalue == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    memcpy(attrp->xvalue, gbp->pos, (size_t)sz);
    gbp->pos = (void *)((char *)(gbp->pos) + sz);

    return 0;
}

static int deserialize_attr(metabuffer *gbp, hdr_attr *attrp) {
    uint32_t tmp;
    int err;
    char *name;
    err = deserialize_name(gbp, &name);
    if (err != NC_NOERR) return err;
    attrp->name = name;
    attrp->name_len = strlen(name);
    err = deserialize_nc_type(gbp, &attrp->xtype);
    if (err != NC_NOERR) return err;
    err = get_uint32((void**)&gbp->pos, &tmp);
    attrp->nelems = (int)tmp;
    if (err != NC_NOERR) return err;
    err = deserialize_attrV(gbp, attrp);
    if (err != NC_NOERR) return err;

    return 0;
}

static int deserialize_attrarray(metabuffer *gbp, hdr_attrarray *ncap) {
    unsigned int tag;
    get_uint32((void**)&gbp->pos, &tag);
    uint32_t tmp;

    if (tag == NC_UNSPECIFIED) {
        get_uint32((void**)&gbp->pos, &tmp);
        ncap->ndefined = (int) tmp;
        assert(ncap->ndefined == 0);
        return 0; // ABSENT
    } else if (tag == NC_ATTRIBUTE) {
        get_uint32((void**)&gbp->pos, &tmp);
        ncap->ndefined = (int) tmp;
        ncap->value = (hdr_attr **)malloc(ncap->ndefined * sizeof(hdr_attr *));
        if (ncap->value == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return -1;
        }
        for (int i = 0; i < ncap->ndefined; i++) {
            ncap->value[i] = (hdr_attr *)malloc(sizeof(hdr_attr));
            if (ncap->value[i] == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                return -1;
            }

            if (deserialize_attr(gbp, ncap->value[i]) != 0) {
                return -1;
            }
        }
    }

    return 0;
}

static int deserialize_var(metabuffer *gbp, hdr_var *varp) {
    int err;
    char *name;
    // if (deserialize_name(gbp, &varp->name) != 0) {
    //     return -1;
    // }
    /* get name */
    err = deserialize_name(gbp, &name);
    if (err != NC_NOERR) return err;
    varp->name = name;
    varp->name_len = strlen(name);
    /* nelems (number of dimensions) */
    u_int32_t tmp;
    get_uint32((void**)&gbp->pos, (unsigned int *)&tmp);
    varp->ndims = (int) tmp;
    varp->dimids = (int *)malloc(varp->ndims * sizeof(int));
    if (varp->dimids == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    for (int i = 0; i < varp->ndims; i++) {
        get_uint32((void**)&gbp->pos, &tmp);
        varp->dimids[i] = (int)tmp;
    }

    if (deserialize_attrarray(gbp, &varp->attrs) != 0) {
        return -1;
    }
    err = deserialize_nc_type(gbp, &varp->xtype);
    if (err != NC_NOERR) return err;

    return 0;
}

static int deserialize_vararray(metabuffer *gbp, hdr_vararray *ncap) {
    unsigned int tag;
    int err;
    uint32_t tmp;
    get_uint32((void**)&gbp->pos, &tag);

    if (tag == NC_UNSPECIFIED) {
        get_uint32((void**)&gbp->pos, (unsigned int *)&tmp);
        ncap->ndefined = (int)tmp;
        assert(ncap->ndefined == 0);
        return 0; // ABSENT
    } else if (tag == NC_VARIABLE) {
        get_uint32((void**)&gbp->pos, (unsigned int *)&tmp);
        ncap->ndefined = (int)tmp;
        ncap->value = (hdr_var **)malloc(ncap->ndefined * sizeof(hdr_var *));
        if (ncap->value == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return -1;
        }

        for (int i = 0; i < ncap->ndefined; i++) {
            ncap->value[i] = (hdr_var *)malloc(sizeof(hdr_var));
            if (ncap->value[i] == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                return -1;
            }
            if (deserialize_var(gbp, ncap->value[i]) != 0) {
                return -1;
            }
        }
    }

    return 0;
}

int deserialize_hdr(struct hdr *ncp, void *buf, int buf_size) {

    int status;
    metabuffer getbuf;

    getbuf.pos           = buf;
    getbuf.base          = buf;
    getbuf.size          = buf_size;

    /* get dim_list from getbuf into ncp */
    status = deserialize_dimarray(&getbuf, &ncp->dims);
    if (status != NC_NOERR) return status;
    

    status = deserialize_vararray(&getbuf, &ncp->vars);
    if (status != NC_NOERR) return status;
    // printf("HERE: %ld", getbuf.pos - getbuf.base);
    assert((int)(getbuf.pos - getbuf.base) == getbuf.size);



    return 0;
}
