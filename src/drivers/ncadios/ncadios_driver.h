/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

#ifndef _ncadios_DRIVER_H
#define _ncadios_DRIVER_H

#include <mpi.h>
#include <pnetcdf.h>
#include <dispatch.h>
#include <adios_read.h>
#include <adios_error.h>


typedef struct NC_ad_dim {
    int len;
    char* name;
} NC_ad_dim;

typedef struct NC_ad_dim_list {
    NC_ad_dim *data;
    int cnt;
    int nalloc;
} NC_ad_dim_list;

/*
typedef struct NC_ad_att {
    char *name;
    nc_type type;
    MPI_Offset len;
    void* data;
} NC_ad_att;
*/

typedef struct NC_ad_att_list {
    int *data;
    int cnt;
    int nalloc;
} NC_ad_att_list;

typedef struct NC_ad_var {
    char *name;
    int ndim;
    int *dimids;
    nc_type type;
    NC_ad_att_list atts;
} NC_ad_var;

typedef struct NC_ad_var_list {
    NC_ad_var *data;
    int cnt;
    int nalloc;
} NC_ad_var_list;

typedef struct NC_ad NC_ad; /* forward reference */
struct NC_ad {
    int                mode;        /* file _open/_create mode */
    int                flag;        /* define/data/collective/indep mode */
    char              *path;        /* path name */
    MPI_Comm           comm;        /* MPI communicator */
    ADIOS_FILE          *fp;        /* ADIOS file pointer */
    int              *ndims;        /* Number of dims in each var */
    int                rank;
    MPI_Offset         nrec;        // Number of records in unlimited dimension
    int              recdim;        // ID of unlimited dimension
    NC_ad_var_list     vars;
    NC_ad_att_list     atts;
    NC_ad_dim_list     dims;
};

extern int 
ncadiosi_parse_header_bp2ncd (NC_ad *ncid);

extern int 
ncadiosi_parse_header_readall (NC_ad *ncadp);

extern int 
ncadiosi_parse_rec_dim(NC_ad *ncadp);

extern int 
ncadiosi_var_list_init(NC_ad_var_list *list);

extern int 
ncadiosi_dim_list_init(NC_ad_dim_list *list);

extern int 
ncadiosi_att_list_init(NC_ad_att_list *list);

extern int 
ncadiosi_var_list_free(NC_ad_var_list *list);

extern int 
ncadiosi_dim_list_free(NC_ad_dim_list *list);

extern int 
ncadiosi_att_list_free(NC_ad_att_list *list);

extern int 
ncadiosi_var_list_add(NC_ad_var_list *list, NC_ad_var data);

extern int 
ncadiosi_att_list_add(NC_ad_att_list *list, int data);

extern int 
ncadiosi_dim_list_add(NC_ad_dim_list *list, NC_ad_dim data);

extern int 
ncadiosi_var_list_find(NC_ad_var_list *list, char *name);

extern int 
ncadiosi_att_list_find(NC_ad_att_list *list, int data);

extern int 
ncadiosi_dim_list_find(NC_ad_dim_list *list, char *name);

extern int 
ncadiosi_inq_varid(NC_ad* ncadp, char* name, int *id);

extern int 
ncadiosi_inq_attid(NC_ad* ncadp, int vid, char* name, int *id);

extern int 
ncadiosi_inq_dimid(NC_ad* ncadp, char* name, int *id);

extern int 
ncadiosi_def_var(NC_ad* ncadp, char* name, nc_type type, int ndim, int *dimids, int *id);

extern int 
ncadiosi_def_dim(NC_ad* ncadp, char* name, int len, int *id);

extern int 
ncadios_sync_header(NC_ad *ncadp);

extern int 
ncadiosi_parse_attrs(NC_ad *ncadp);

extern int 
ncadiosiconvert(void *inbuf, void *outbuf, MPI_Datatype intype, MPI_Datatype outtype, int N);

extern nc_type 
ncadios_to_nc_type(enum ADIOS_DATATYPES atype);

extern MPI_Datatype 
ncadios_to_mpi_type(enum ADIOS_DATATYPES atype);

extern int
ncadios_create(MPI_Comm comm, const char *path, int cmode, int ncid, MPI_Info info, void **ncdp);

extern int
ncadios_open(MPI_Comm comm, const char *path, int omode, int ncid, MPI_Info info, void **ncdp);

extern int
ncadios_close(void *ncdp);

extern int
ncadios_enddef(void *ncdp);

extern int
ncadios__enddef(void *ncdp, MPI_Offset h_minfree, MPI_Offset v_align, MPI_Offset v_minfree, MPI_Offset r_align);

extern int
ncadios_redef(void *ncdp);

extern int
ncadios_sync(void *ncdp);

extern int
ncadios_flush(void *ncdp);

extern int
ncadios_abort(void *ncdp);

extern int
ncadios_set_fill(void *ncdp, int fill_mode, int *old_fill_mode);

extern int
ncadios_fill_var_rec(void *ncdp, int varid, MPI_Offset recno);

extern int
ncadios_inq(void *ncdp, int *ndimsp, int *nvarsp, int *nattsp, int *xtendimp);

extern int
ncadios_inq_misc(void *ncdp, int *pathlen, char *path, int *num_fix_varsp,
               int *num_rec_varsp, int *striping_size, int *striping_count,
               MPI_Offset *header_size, MPI_Offset *header_extent,
               MPI_Offset *recsize, MPI_Offset *put_size, MPI_Offset *get_size,
               MPI_Info *info_used, int *nreqs, MPI_Offset *usage,
               MPI_Offset *buf_size);

extern int
ncadios_sync_numrecs(void *ncdp);

extern int
ncadios_begin_indep_data(void *ncdp);

extern int
ncadios_end_indep_data(void *ncdp);

extern int
ncadios_def_dim(void *ncdp, const char *name, MPI_Offset size, int *dimidp);

extern int
ncadios_inq_dimid(void *ncdp, const char *name, int *dimidp);

extern int
ncadios_inq_dim(void *ncdp, int dimid, char *name, MPI_Offset *lengthp);

extern int
ncadios_rename_dim(void *ncdp, int dimid, const char *newname);

extern int
ncadios_inq_att(void *ncdp, int varid, const char *name, nc_type *xtypep, MPI_Offset *lenp);

extern int
ncadios_inq_attid(void *ncdp, int varid, const char *name, int *idp);

extern int
ncadios_inq_attname(void *ncdp, int varid, int attnum, char *name);

extern int
ncadios_copy_att(void *ncdp_in, int varid_in, const char *name, void *ncdp_out, int varid_out);

extern int
ncadios_rename_att(void *ncdp, int varid, const char *name, const char *newname);

extern int
ncadios_del_att(void *ncdp, int varid, const char *name);

extern int
ncadios_get_att(void *ncdp, int varid, const char *name, void *value, MPI_Datatype itype);

extern int
ncadios_put_att(void *ncdp, int varid, const char *name, nc_type xtype, MPI_Offset nelems, const void *value, MPI_Datatype itype);

extern int
ncadios_def_var(void *ncdp, const char *name, nc_type type, int ndims, const int *dimids, int *varidp);

extern int
ncadios_def_var_fill(void *ncdp, int varid, int nofill, const void *fill_value);

extern int
ncadios_inq_var(void *ncdp, int varid, char *name, nc_type *xtypep, int *ndimsp,
               int *dimids, int *nattsp, MPI_Offset *offsetp, int *no_fill, void *fill_value);

extern int
ncadios_inq_varid(void *ncdp, const char *name, int *varid);

extern int
ncadios_rename_var(void *ncdp, int varid, const char *newname);

extern int
ncadios_get_var(void *ncdp, int varid, const MPI_Offset *start, const MPI_Offset *count, const MPI_Offset *stride, const MPI_Offset *imap, void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int reqMode);

extern int
ncadios_put_var(void *ncdp, int varid, const MPI_Offset *start, const MPI_Offset *count, const MPI_Offset *stride, const MPI_Offset *imap, const void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int reqMode);

extern int
ncadios_get_varn(void *ncdp, int varid, int num, MPI_Offset* const *starts, MPI_Offset* const *counts, void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int reqMode);

extern int
ncadios_put_varn(void *ncdp, int varid, int num, MPI_Offset* const *starts, MPI_Offset* const *counts, const void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int reqMode);

extern int
ncadios_get_vard(void *ncdp, int varid, MPI_Datatype filetype, void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int reqMode);

extern int
ncadios_put_vard(void *ncdp, int varid, MPI_Datatype filetype, const void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int reqMode);

extern int
ncadios_iget_var(void *ncdp, int varid, const MPI_Offset *start, const MPI_Offset *count, const MPI_Offset *stride, const MPI_Offset *imap, void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int *req, int reqMode);

extern int
ncadios_iput_var(void *ncdp, int varid, const MPI_Offset *start, const MPI_Offset *count, const MPI_Offset *stride, const MPI_Offset *imap, const void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int *req, int reqMode);

extern int
ncadios_bput_var(void *ncdp, int varid, const MPI_Offset *start, const MPI_Offset *count, const MPI_Offset *stride, const MPI_Offset *imap, const void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int *req, int reqMode);

extern int
ncadios_iget_varn(void *ncdp, int varid, int num, MPI_Offset* const *starts, MPI_Offset* const *counts, void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int *reqid, int reqMode);

extern int
ncadios_iput_varn(void *ncdp, int varid, int num, MPI_Offset* const *starts, MPI_Offset* const *counts, const void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int *reqid, int reqMode);

extern int
ncadios_bput_varn(void *ncdp, int varid, int num, MPI_Offset* const *starts, MPI_Offset* const *counts, const void *buf, MPI_Offset bufcount, MPI_Datatype buftype, int *reqid, int reqMode);

extern int
ncadios_buffer_attach(void *ncdp, MPI_Offset bufsize);

extern int
ncadios_buffer_detach(void *ncdp);

extern int
ncadios_wait(void *ncdp, int num_reqs, int *req_ids, int *statuses, int reqMode);

extern int
ncadios_cancel(void *ncdp, int num_reqs, int *req_ids, int *statuses);

#endif
