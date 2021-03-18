/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the following PnetCDF APIs.
 *
 * ncmpi_def_var()                  : dispatcher->def_var()
 * ncmpi_inq_varid()                : dispatcher->inq_varid()
 * ncmpi_inq_var()                  : dispatcher->inq_var()
 * ncmpi_rename_var()               : dispatcher->rename_var()
 *
 * ncmpi_get_var<kind>()            : dispatcher->get_var()
 * ncmpi_put_var<kind>()            : dispatcher->put_var()
 * ncmpi_get_var<kind>_<type>()     : dispatcher->get_var()
 * ncmpi_put_var<kind>_<type>()     : dispatcher->put_var()
 * ncmpi_get_var<kind>_all()        : dispatcher->get_var()
 * ncmpi_put_var<kind>_all()        : dispatcher->put_var()
 * ncmpi_get_var<kind>_<type>_all() : dispatcher->get_var()
 * ncmpi_put_var<kind>_<type>_all() : dispatcher->put_var()
 *
 * ncmpi_iget_var<kind>()           : dispatcher->iget_var()
 * ncmpi_iput_var<kind>()           : dispatcher->iput_var()
 * ncmpi_iget_var<kind>_<type>()    : dispatcher->iget_var()
 * ncmpi_iput_var<kind>_<type>()    : dispatcher->iput_var()
 *
 * ncmpi_buffer_attach()            : dispatcher->buffer_attach()
 * ncmpi_buffer_detach()            : dispatcher->buffer_detach()
 * ncmpi_bput_var<kind>_<type>()    : dispatcher->bput_var()
 *
 * ncmpi_get_varn_<type>()          : dispatcher->get_varn()
 * ncmpi_put_varn_<type>()          : dispatcher->put_varn()
 *
 * ncmpi_iget_varn_<type>()         : dispatcher->iget_varn()
 * ncmpi_iput_varn_<type>()         : dispatcher->iput_varn()
 * ncmpi_bput_varn_<type>()         : dispatcher->bput_varn()
 *
 * ncmpi_get_vard()                 : dispatcher->get_vard()
 * ncmpi_put_vard()                 : dispatcher->put_vard()
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <common.h>
#include <mpi.h>
#include <nczipio_driver.h>
#include <pnc_debug.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ncmpio/ncmpio_NC.h"
#include "nczipio_internal.h"

int nczipio_def_var (
	void *ncdp, const char *name, nc_type xtype, int ndims, const int *dimids, int *varidp) {
	int i, err;
	NC_zip *nczipp = (NC_zip *)ncdp;
	NC_zip_var *varp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)
	NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_INIT)

	err = nczipioi_var_list_add (&(nczipp->vars));
	if (err < 0) return err;
	*varidp = err;

	varp = nczipp->vars.data + (*varidp);

	varp->ndim		  = ndims;
	varp->chunkdim	  = NULL;
	varp->chunk_index = NULL;
	varp->chunk_owner = NULL;
	varp->xtype		  = xtype;
	varp->esize		  = NC_Type_size (xtype);
	varp->etype		  = ncmpii_nc2mpitype (xtype);
	varp->isnew		  = 1;
	varp->expanded	  = 0;

	if (ndims < 1) {  // Do not compress scalar
		varp->varkind = NC_ZIP_VAR_RAW;
		varp->dimsize = NULL;

		err = nczipp->driver->def_var (nczipp->ncp, name, xtype, ndims, dimids, &varp->varid);
		if (err != NC_NOERR) return err;

		err = nczipp->driver->put_att (nczipp->ncp, varp->varid, "_varkind", NC_INT, 1,
									   &(varp->varkind), MPI_INT);	// Comressed var?
		if (err != NC_NOERR) return err;
	} else {
		err = nczipp->driver->def_var (nczipp->ncp, name, xtype, 0, NULL,
									   &varp->varid);  // Dummy var for attrs
		if (err != NC_NOERR) return err;

		varp->varkind = NC_ZIP_VAR_COMPRESSED;
		varp->dimids  = (int *)NCI_Malloc (sizeof (int) * ndims);
		memcpy (varp->dimids, dimids, sizeof (int) * ndims);
		varp->dimsize = (MPI_Offset *)NCI_Malloc (sizeof (MPI_Offset) * ndims);
		for (i = 0; i < ndims; i++) {
			nczipp->driver->inq_dim (nczipp->ncp, dimids[i], NULL, varp->dimsize + i);
		}
		if (varp->dimids[0] == nczipp->recdim) {
			varp->isrec = 1;
		} else {
			varp->isrec = 0;
		}

		err = nczipp->driver->put_att (nczipp->ncp, varp->varid, "_ndim", NC_INT, 1, &ndims,
									   MPI_INT);  // Original dimensions
		if (err != NC_NOERR) return err;
		err = nczipp->driver->put_att (nczipp->ncp, varp->varid, "_dimids", NC_INT, ndims, dimids,
									   MPI_INT);  // Dimensiona IDs
		if (err != NC_NOERR) return err;
		err = nczipp->driver->put_att (nczipp->ncp, varp->varid, "_datatype", NC_INT, 1, &xtype,
									   MPI_INT);  // Original datatype
		if (err != NC_NOERR) return err;
		err = nczipp->driver->put_att (nczipp->ncp, varp->varid, "_varkind", NC_INT, 1,
									   &(varp->varkind), MPI_INT);	// Comressed var?
		if (err != NC_NOERR) return err;
	}

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_INIT)
	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	return NC_NOERR;
}

int nczipio_inq_varid (void *ncdp, const char *name, int *varid) {
	int i, vid, err;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)

	err = nczipp->driver->inq_varid (nczipp->ncp, name, &vid);
	if (err != NC_NOERR) return err;

	if (varid != NULL) {
		for (i = 0; i < nczipp->vars.cnt; i++) {
			if (nczipp->vars.data[i].varid == vid) {
				*varid = i;
				break;
			}
		}
		if (i >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_ENOTVAR) }
	}

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	return NC_NOERR;
}

int nczipio_inq_var (void *ncdp,
					 int varid,
					 char *name,
					 nc_type *xtypep,
					 int *ndimsp,
					 int *dimids,
					 int *nattsp,
					 MPI_Offset *offsetp,
					 int *no_fillp,
					 void *fill_valuep) {
	int err;
	NC_zip *nczipp = (NC_zip *)ncdp;
	NC_zip_var *varp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }

	varp = nczipp->vars.data + varid;

	err = nczipp->driver->inq_var (nczipp->ncp, varp->varid, name, xtypep, NULL, NULL, nattsp,
								   offsetp, no_fillp, fill_valuep);
	if (err != NC_NOERR) return err;

	if (ndimsp != NULL) { *ndimsp = varp->ndim; }

	if (dimids != NULL) { memcpy (dimids, varp->dimids, sizeof (int) * varp->ndim); }

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	return NC_NOERR;
}

int nczipio_rename_var (void *ncdp, int varid, const char *newname) {
	int err;
	NC_zip *nczipp = (NC_zip *)ncdp;
	NC_zip_var *varp;

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	err = nczipp->driver->rename_var (nczipp->ncp, varp->varid, newname);
	if (err != NC_NOERR) return err;

	return NC_NOERR;
}

int nczipio_get_var (void *ncdp,
					 int varid,
					 const MPI_Offset *start,
					 const MPI_Offset *count,
					 const MPI_Offset *stride,
					 const MPI_Offset *imap,
					 void *buf,
					 MPI_Offset bufcount,
					 MPI_Datatype buftype,
					 int reqMode) {
	int err = NC_NOERR, status = NC_NOERR;
	void *cbuf = (void *)buf;
	void *xbuf = (void *)buf;
	MPI_Offset nelem;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)
	NC_ZIP_TIMER_START (NC_ZIP_TIMER_GET)

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		return nczipp->driver->get_var (nczipp->ncp, varp->varid, start, count, stride, imap, buf,
										bufcount, buftype, reqMode);
	}

	if (nczipp->delay_init && (varp->chunkdim == NULL)) {
		NC_ZIP_TIMER_PAUSE (NC_ZIP_TIMER_GET)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_INIT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_INIT_META)

		err = nczipioi_var_init (nczipp, varp, 1, (MPI_Offset **)&start, (MPI_Offset **)&count);
		CHK_ERR

		if (!(varp->isnew)) {
			err = nczipp->driver->get_att (nczipp->ncp, varp->varid, "_metaoffset",
										   &(varp->metaoff), MPI_LONG_LONG);
			if (err == NC_NOERR) {	// Read index table
				MPI_Status status;

				// Set file view
				CHK_ERR_SET_VIEW (((NC *)(nczipp->ncp))->collective_fh,
								  ((NC *)(nczipp->ncp))->begin_var, MPI_BYTE, MPI_BYTE, "native",
								  MPI_INFO_NULL);
				// Read data
				CHK_ERR_READ_AT_ALL (
					((NC *)(nczipp->ncp))->collective_fh, varp->metaoff, varp->chunk_index,
					sizeof (NC_zip_chunk_index_entry) * varp->nchunk, MPI_BYTE, &status);
			} else {
				varp->metaoff = -1;
				memset (varp->chunk_index, 0,
						sizeof (NC_zip_chunk_index_entry) * (varp->nchunk + 1));
			}
		}

		NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_INIT_META)
		NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_INIT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_GET)
	}

	if (varp->isrec && (varp->dimsize[0] < nczipp->recsize) &&
		(start[0] + count[0] >= varp->dimsize[0])) {
		NC_ZIP_TIMER_PAUSE (NC_ZIP_TIMER_GET)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_RESIZE)

		err = nczipioi_var_resize (nczipp, varp);
		CHK_ERR

		NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_RESIZE)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_GET)
	}

	if (buftype != varp->etype) {
		int i;

		nelem = 1;
		for (i = 0; i < varp->ndim; i++) { nelem *= count[i]; }

		xbuf = (char *)NCI_Malloc (nelem * varp->esize);
		CHK_PTR (xbuf)
	} else {
		xbuf = cbuf;
	}

	// Collective buffer
	switch (nczipp->comm_unit) {
		case NC_ZIP_COMM_CHUNK:
			err = nczipioi_get_var_cb_chunk (nczipp, varp, start, count, stride, xbuf);
			break;
		case NC_ZIP_COMM_PROC:
			err = nczipioi_get_var_cb_proc (nczipp, varp, start, count, stride, xbuf);
			break;
	}
	CHK_ERR

	if (buftype != varp->etype) {
		err = nczipioiconvert (xbuf, cbuf, varp->etype, buftype, nelem);
		if (err != NC_NOERR) return err;
	}

	if (xbuf != cbuf) NCI_Free (xbuf);
	if (cbuf != buf) NCI_Free (cbuf);

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_GET)
	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

err_out:;
	if (status == NC_NOERR) status = err;
	return status; /* first error encountered */
}

int nczipio_put_var (void *ncdp,
					 int varid,
					 const MPI_Offset *start,
					 const MPI_Offset *count,
					 const MPI_Offset *stride,
					 const MPI_Offset *imap,
					 const void *buf,
					 MPI_Offset bufcount,
					 MPI_Datatype buftype,
					 int reqMode) {
	int err	   = NC_NOERR;
	void *cbuf = (void *)buf;
	void *xbuf = (void *)buf;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)
	NC_ZIP_TIMER_START (NC_ZIP_TIMER_PUT)

	if (reqMode == NC_REQ_INDEP) { DEBUG_RETURN_ERROR (NC_ENOTSUPPORT); }

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		return nczipp->driver->put_var (nczipp->ncp, varp->varid, start, count, stride, imap, buf,
										bufcount, buftype, reqMode);
	}

	if (nczipp->delay_init && (varp->chunkdim == NULL)) {
		NC_ZIP_TIMER_PAUSE (NC_ZIP_TIMER_PUT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_INIT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_INIT_META)

		err = nczipioi_var_init (nczipp, varp, 1, (MPI_Offset **)&start, (MPI_Offset **)&count);
		CHK_ERR

		if (!(varp->isnew)) {
			err = nczipp->driver->get_att (nczipp->ncp, varp->varid, "_metaoffset",
										   &(varp->metaoff), MPI_LONG_LONG);
			if (err == NC_NOERR) {	// Read index table
				MPI_Status status;

				// Set file view
				CHK_ERR_SET_VIEW (((NC *)(nczipp->ncp))->collective_fh,
								  ((NC *)(nczipp->ncp))->begin_var, MPI_BYTE, MPI_BYTE, "native",
								  MPI_INFO_NULL);
				// Read data
				CHK_ERR_READ_AT_ALL (
					((NC *)(nczipp->ncp))->collective_fh, varp->metaoff, varp->chunk_index,
					sizeof (NC_zip_chunk_index_entry) * varp->nchunk, MPI_BYTE, &status);
			} else {
				varp->metaoff = -1;
				memset (varp->chunk_index, 0,
						sizeof (NC_zip_chunk_index_entry) * (varp->nchunk + 1));
			}
		}

		NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_INIT_META)
		NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_INIT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_PUT)
	}

	if (imap != NULL || bufcount != -1) {
		/* pack buf to cbuf -------------------------------------------------*/
		/* If called from a true varm API or a flexible API, ncmpii_pack()
		 * packs user buf into a contiguous cbuf (need to be freed later).
		 * Otherwise, cbuf is simply set to buf. ncmpii_pack() also returns
		 * etype (MPI primitive datatype in buftype), and nelems (number of
		 * etypes in buftype * bufcount)
		 */
		int ndims;
		MPI_Offset nelems;
		MPI_Datatype etype;

		err = nczipp->driver->inq_var (nczipp->ncp, varid, NULL, NULL, &ndims, NULL, NULL, NULL,
									   NULL, NULL);
		if (err != NC_NOERR) goto err_check;

		err = ncmpii_pack (ndims, count, imap, (void *)buf, bufcount, buftype, &nelems, &etype,
						   &cbuf);
		if (err != NC_NOERR) goto err_check;

		imap	 = NULL;
		bufcount = (nelems == 0) ? 0 : -1; /* make it a high-level API */
		buftype	 = etype;				   /* an MPI primitive type */
	}

err_check:
	if (err != NC_NOERR) {
		if (reqMode & NC_REQ_INDEP) return err;
		reqMode |= NC_REQ_ZERO; /* participate collective call */
	}

	if (buftype != varp->etype) {
		int i;
		MPI_Offset nelem;

		nelem = 1;
		for (i = 0; i < varp->ndim; i++) { nelem *= count[i]; }

		xbuf = (char *)NCI_Malloc (nelem * varp->esize);
		CHK_PTR (xbuf)
		err = nczipioiconvert (cbuf, xbuf, buftype, varp->etype, nelem);
		if (err != NC_NOERR) return err;
	} else {
		xbuf = cbuf;
	}

	err = nczipioi_put_var (nczipp, varp, start, count, stride, xbuf);
	CHK_ERR

	if (cbuf != buf) NCI_Free (cbuf);

	if (xbuf != cbuf) NCI_Free (xbuf);
	if (cbuf != buf) NCI_Free (cbuf);

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_PUT)
	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

err_out:;
	return err; /* first error encountered */
}

int nczipio_iget_var (void *ncdp,
					  int varid,
					  const MPI_Offset *start,
					  const MPI_Offset *count,
					  const MPI_Offset *stride,
					  const MPI_Offset *imap,
					  void *buf,
					  MPI_Offset bufcount,
					  MPI_Datatype buftype,
					  int *reqid,
					  int reqMode) {
	int err;
	void *cbuf = (void *)buf;
	void *xbuf = (void *)buf;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_IGET)

	if (reqMode == NC_REQ_INDEP) { DEBUG_RETURN_ERROR (NC_ENOTSUPPORT); }

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		err = nczipp->driver->iget_var (nczipp->ncp, varp->varid, start, count, stride, imap, buf,
										bufcount, buftype, reqid, reqMode);
		if (err != NC_NOERR) { return err; }
		if (reqid != NULL) { *reqid = *reqid * 2 + 1; }
		return NC_NOERR;
	}

	nczipioi_iget_var (nczipp, varid, start, count, stride, imap, buf, bufcount, buftype, reqid);
	if (reqid != NULL) { (*reqid) *= 2; }

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_IGET)

	return NC_NOERR;
}

int nczipio_iput_var (void *ncdp,
					  int varid,
					  const MPI_Offset *start,
					  const MPI_Offset *count,
					  const MPI_Offset *stride,
					  const MPI_Offset *imap,
					  const void *buf,
					  MPI_Offset bufcount,
					  MPI_Datatype buftype,
					  int *reqid,
					  int reqMode) {
	int err	   = NC_NOERR;
	void *cbuf = (void *)buf;
	void *xbuf = (void *)buf;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_IPUT)

	if (reqMode == NC_REQ_INDEP) { DEBUG_RETURN_ERROR (NC_ENOTSUPPORT); }

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		err = nczipp->driver->iput_var (nczipp->ncp, varp->varid, start, count, stride, imap, buf,
										bufcount, buftype, reqid, reqMode);
		if (err != NC_NOERR) { return err; }
		if (reqid != NULL) { *reqid = *reqid * 2 + 1; }
		return NC_NOERR;
	}

	if (varp->isrec) {
		if (nczipp->recsize < start[0] + count[0]) { nczipp->recsize = start[0] + count[0]; }
	}

	if (imap != NULL || bufcount != -1) {
		/* pack buf to cbuf -------------------------------------------------*/
		/* If called from a true varm API or a flexible API, ncmpii_pack()
		 * packs user buf into a contiguous cbuf (need to be freed later).
		 * Otherwise, cbuf is simply set to buf. ncmpii_pack() also returns
		 * etype (MPI primitive datatype in buftype), and nelems (number of
		 * etypes in buftype * bufcount)
		 */
		int ndims;
		MPI_Offset nelems;
		MPI_Datatype etype;

		err = nczipp->driver->inq_var (nczipp->ncp, varid, NULL, NULL, &ndims, NULL, NULL, NULL,
									   NULL, NULL);
		if (err != NC_NOERR) goto err_check;

		err = ncmpii_pack (ndims, count, imap, (void *)buf, bufcount, buftype, &nelems, &etype,
						   &cbuf);
		if (err != NC_NOERR) goto err_check;

		imap	 = NULL;
		bufcount = (nelems == 0) ? 0 : -1; /* make it a high-level API */
		buftype	 = etype;				   /* an MPI primitive type */
	}

err_check:
	if (err != NC_NOERR) {
		if (reqMode & NC_REQ_INDEP) return err;
		reqMode |= NC_REQ_ZERO; /* participate collective call */
	}

	if (buftype != varp->etype) {
		int i;
		MPI_Offset nelem;

		nelem = 1;
		for (i = 0; i < varp->ndim; i++) { nelem *= count[i]; }

		xbuf = (char *)NCI_Malloc (nelem * varp->esize);
		CHK_PTR (xbuf)
		err = nczipioiconvert (cbuf, xbuf, buftype, varp->etype, nelem);
		if (err != NC_NOERR) return err;
	} else {
		xbuf = cbuf;
	}

	err = nczipioi_iput_var (nczipp, varid, start, count, stride, xbuf, buf, reqid);
	if (reqid != NULL) { (*reqid) *= 2; }

	if (cbuf != buf && cbuf != xbuf) NCI_Free (cbuf);

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_IPUT)

err_out:;
	return err;
}

int nczipio_buffer_attach (void *ncdp, MPI_Offset bufsize) {
	int err;
	NC_zip *nczipp = (NC_zip *)ncdp;

	err = nczipp->driver->buffer_attach (nczipp->ncp, bufsize);
	if (err != NC_NOERR) return err;

	return NC_NOERR;
}

int nczipio_buffer_detach (void *ncdp) {
	int err;
	NC_zip *nczipp = (NC_zip *)ncdp;

	err = nczipp->driver->buffer_detach (nczipp->ncp);
	if (err != NC_NOERR) return err;

	return NC_NOERR;
}

int nczipio_bput_var (void *ncdp,
					  int varid,
					  const MPI_Offset *start,
					  const MPI_Offset *count,
					  const MPI_Offset *stride,
					  const MPI_Offset *imap,
					  const void *buf,
					  MPI_Offset bufcount,
					  MPI_Datatype buftype,
					  int *reqid,
					  int reqMode) {
	int err = NC_NOERR;
	int i;
	void *cbuf = (void *)buf;
	void *xbuf;
	MPI_Offset nelem;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_IPUT)

	if (reqMode == NC_REQ_INDEP) { DEBUG_RETURN_ERROR (NC_ENOTSUPPORT); }

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		err = nczipp->driver->bput_var (nczipp->ncp, varp->varid, start, count, stride, imap, buf,
										bufcount, buftype, reqid, reqMode);
		if (err != NC_NOERR) { return err; }
		if (reqid != NULL) { *reqid = *reqid * 2 + 1; }
		return NC_NOERR;
	}

	if (varp->isrec) {
		if (nczipp->recsize < start[0] + count[0]) { nczipp->recsize = start[0] + count[0]; }
	}

	if (imap != NULL || bufcount != -1) {
		/* pack buf to cbuf -------------------------------------------------*/
		/* If called from a true varm API or a flexible API, ncmpii_pack()
		 * packs user buf into a contiguous cbuf (need to be freed later).
		 * Otherwise, cbuf is simply set to buf. ncmpii_pack() also returns
		 * etype (MPI primitive datatype in buftype), and nelems (number of
		 * etypes in buftype * bufcount)
		 */
		int ndims;
		MPI_Offset nelems;
		MPI_Datatype etype;

		err = nczipp->driver->inq_var (nczipp->ncp, varid, NULL, NULL, &ndims, NULL, NULL, NULL,
									   NULL, NULL);
		if (err != NC_NOERR) goto err_check;

		err = ncmpii_pack (ndims, count, imap, (void *)buf, bufcount, buftype, &nelems, &etype,
						   &cbuf);
		if (err != NC_NOERR) goto err_check;

		imap	 = NULL;
		bufcount = (nelems == 0) ? 0 : -1; /* make it a high-level API */
		buftype	 = etype;				   /* an MPI primitive type */
	}

err_check:
	if (err != NC_NOERR) {
		if (reqMode & NC_REQ_INDEP) return err;
		reqMode |= NC_REQ_ZERO; /* participate collective call */
	}

	nelem = 1;
	for (i = 0; i < varp->ndim; i++) { nelem *= count[i]; }

	xbuf = (char *)NCI_Malloc (nelem * varp->esize);
	CHK_PTR (xbuf)

	if (buftype != varp->etype) {
		err = nczipioiconvert (cbuf, xbuf, buftype, varp->etype, nelem);
		if (err != NC_NOERR) return err;
	} else {
		memcpy (xbuf, cbuf, varp->esize * nelem);
	}

	err = nczipioi_iput_var (nczipp, varid, start, count, stride, xbuf, buf, reqid);
	CHK_ERR
	if (reqid != NULL) { (*reqid) *= 2; }

	if (cbuf != buf) NCI_Free (cbuf);

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_IPUT)

err_out:;
	return err;
}
int nczipio_get_varn (void *ncdp,
					  int varid,
					  int num,
					  MPI_Offset *const *starts,
					  MPI_Offset *const *counts,
					  void *buf,
					  MPI_Offset bufcount,
					  MPI_Datatype buftype,
					  int reqMode) {
	int err = NC_NOERR;
	int i;
	void *cbuf = (void *)buf;
	void *xbuf = (void *)buf;
	MPI_Offset nelem;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)
	NC_ZIP_TIMER_START (NC_ZIP_TIMER_GET)

	if (reqMode == NC_REQ_INDEP) { DEBUG_RETURN_ERROR (NC_ENOTSUPPORT); }

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		return nczipp->driver->get_varn (nczipp->ncp, varp->varid, num, starts, counts, buf,
										 bufcount, buftype, reqMode);
	}

	if (nczipp->delay_init && (varp->chunkdim == NULL)) {
		NC_ZIP_TIMER_PAUSE (NC_ZIP_TIMER_GET)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_INIT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_INIT_META)

		err = nczipioi_var_init (nczipp, varp, num, (MPI_Offset **)starts, (MPI_Offset **)counts);
		CHK_ERR

		if (!(varp->isnew)) {
			err = nczipp->driver->get_att (nczipp->ncp, varp->varid, "_metaoffset",
										   &(varp->metaoff), MPI_LONG_LONG);
			if (err == NC_NOERR) {	// Read index table
				MPI_Status status;

				// Set file view
				CHK_ERR_SET_VIEW (((NC *)(nczipp->ncp))->collective_fh,
								  ((NC *)(nczipp->ncp))->begin_var, MPI_BYTE, MPI_BYTE, "native",
								  MPI_INFO_NULL);
				// Read data
				CHK_ERR_READ_AT_ALL (
					((NC *)(nczipp->ncp))->collective_fh, varp->metaoff, varp->chunk_index,
					sizeof (NC_zip_chunk_index_entry) * varp->nchunk, MPI_BYTE, &status);
			} else {
				varp->metaoff = -1;
				memset (varp->chunk_index, 0,
						sizeof (NC_zip_chunk_index_entry) * (varp->nchunk + 1));
			}
		}

		NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_INIT_META)
		NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_INIT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_GET)
	}

	if (varp->isrec && (varp->dimsize[0] < nczipp->recsize)) {
		for (i = 0; i < num; i++) {
			if (starts[i][0] + counts[i][0] >= varp->dimsize[0]) {
				NC_ZIP_TIMER_PAUSE (NC_ZIP_TIMER_GET)
				NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_RESIZE)

				err = nczipioi_var_resize (nczipp, varp);
				CHK_ERR

				NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_RESIZE)
				NC_ZIP_TIMER_START (NC_ZIP_TIMER_GET)

				break;
			}
		}
	}

	if (buftype != varp->etype) {
		int j;
		MPI_Offset tmp;

		nelem = 0;
		for (i = 0; i < num; i++) {
			tmp = 1;
			for (j = 0; j < varp->ndim; j++) { tmp *= counts[i][j]; }
			nelem += tmp;
		}

		xbuf = (char *)NCI_Malloc (nelem * varp->esize);
		CHK_PTR (xbuf)
	} else {
		xbuf = cbuf;
	}

	err = nczipioi_get_varn (nczipp, varp, num, starts, counts, xbuf);
	if (err != NC_NOERR) return err;

	if (buftype != varp->etype) {
		err = nczipioiconvert (xbuf, cbuf, varp->etype, buftype, nelem);
		if (err != NC_NOERR) return err;
	}

	if (xbuf != cbuf) NCI_Free (xbuf);
	if (cbuf != buf) NCI_Free (cbuf);

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_GET)
	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

err_out:;
	return err;
}

int nczipio_put_varn (void *ncdp,
					  int varid,
					  int num,
					  MPI_Offset *const *starts,
					  MPI_Offset *const *counts,
					  const void *buf,
					  MPI_Offset bufcount,
					  MPI_Datatype buftype,
					  int reqMode) {
	int err	   = NC_NOERR;
	void *cbuf = (void *)buf;
	void *xbuf = (void *)buf;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)
	NC_ZIP_TIMER_START (NC_ZIP_TIMER_PUT)

	if (reqMode == NC_REQ_INDEP) { DEBUG_RETURN_ERROR (NC_ENOTSUPPORT); }

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		return nczipp->driver->put_varn (nczipp->ncp, varp->varid, num, starts, counts, buf,
										 bufcount, buftype, reqMode);
	}

	if (nczipp->delay_init && (varp->chunkdim == NULL)) {
		NC_ZIP_TIMER_PAUSE (NC_ZIP_TIMER_PUT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_INIT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_VAR_INIT_META)

		err = nczipioi_var_init (nczipp, varp, num, (MPI_Offset **)starts, (MPI_Offset **)counts);
		CHK_ERR
		if (!(varp->isnew)) {
			err = nczipp->driver->get_att (nczipp->ncp, varp->varid, "_metaoffset",
										   &(varp->metaoff), MPI_LONG_LONG);
			if (err == NC_NOERR) {	// Read index table
				MPI_Status status;

				// Set file view
				CHK_ERR_SET_VIEW (((NC *)(nczipp->ncp))->collective_fh,
								  ((NC *)(nczipp->ncp))->begin_var, MPI_BYTE, MPI_BYTE, "native",
								  MPI_INFO_NULL);
				// Read data
				CHK_ERR_READ_AT_ALL (
					((NC *)(nczipp->ncp))->collective_fh, varp->metaoff, varp->chunk_index,
					sizeof (NC_zip_chunk_index_entry) * varp->nchunk, MPI_BYTE, &status);
			} else {
				varp->metaoff = -1;
				memset (varp->chunk_index, 0,
						sizeof (NC_zip_chunk_index_entry) * (varp->nchunk + 1));
			}
		}

		NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_INIT_META)
		NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_VAR_INIT)
		NC_ZIP_TIMER_START (NC_ZIP_TIMER_PUT)
	}

	if (buftype != varp->etype) {
		int i, j;
		MPI_Offset nelem, tmp;

		nelem = 0;
		for (i = 0; i < num; i++) {
			tmp = 1;
			for (j = 0; j < varp->ndim; j++) { tmp *= counts[i][j]; }
			nelem += tmp;
		}

		xbuf = (char *)NCI_Malloc (nelem * varp->esize);
		CHK_PTR (xbuf)
		err = nczipioiconvert (cbuf, xbuf, buftype, varp->etype, nelem);
		if (err != NC_NOERR) return err;
	} else {
		xbuf = cbuf;
	}

	err = nczipioi_put_varn (nczipp, varp, num, starts, counts, xbuf);
	if (err != NC_NOERR) return err;

err_out:;
	if (xbuf != cbuf) NCI_Free (xbuf);
	if (cbuf != buf) NCI_Free (cbuf);

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_PUT)
	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	return err;
}

int nczipio_iget_varn (void *ncdp,
					   int varid,
					   int num,
					   MPI_Offset *const *starts,
					   MPI_Offset *const *counts,
					   void *buf,
					   MPI_Offset bufcount,
					   MPI_Datatype buftype,
					   int *reqid,
					   int reqMode) {
	int err;
	void *cbuf = (void *)buf;
	void *xbuf = (void *)buf;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_IGET)

	if (reqMode == NC_REQ_INDEP) { DEBUG_RETURN_ERROR (NC_ENOTSUPPORT); }

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		err = nczipp->driver->iget_varn (nczipp->ncp, varp->varid, num, starts, counts, buf,
										 bufcount, buftype, reqid, reqMode);
		if (err != NC_NOERR) { return err; }
		if (reqid != NULL) { *reqid = *reqid * 2 + 1; }
		return NC_NOERR;
	}

	xbuf = cbuf;
	err	 = nczipioi_iget_varn (nczipp, varid, num, starts, counts, buf, bufcount, buftype, reqid);
	if (err != NC_NOERR) return err;
	if (reqid != NULL) { (*reqid) *= 2; }

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_IGET)

	return NC_NOERR;
}

int nczipio_iput_varn (void *ncdp,
					   int varid,
					   int num,
					   MPI_Offset *const *starts,
					   MPI_Offset *const *counts,
					   const void *buf,
					   MPI_Offset bufcount,
					   MPI_Datatype buftype,
					   int *reqid,
					   int reqMode) {
	int err;
	int i;
	void *cbuf = (void *)buf;
	void *xbuf = (void *)buf;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_IPUT)

	if (reqMode == NC_REQ_INDEP) { DEBUG_RETURN_ERROR (NC_ENOTSUPPORT); }

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->isrec) {
		for (i = 0; i < num; i++) {
			if (nczipp->recsize < starts[i][0] + counts[i][0]) {
				nczipp->recsize = starts[i][0] + counts[i][0];
			}
		}
	}

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		err = nczipp->driver->iput_varn (nczipp->ncp, varp->varid, num, starts, counts, buf,
										 bufcount, buftype, reqid, reqMode);
		if (err != NC_NOERR) { return err; }
		if (reqid != NULL) { *reqid = *reqid * 2 + 1; }
		return NC_NOERR;
	}

	if (buftype != varp->etype) {
		int j;
		MPI_Offset nelem, tmp;

		nelem = 0;
		for (i = 0; i < num; i++) {
			tmp = 1;
			for (j = 0; j < varp->ndim; j++) { tmp *= counts[i][j]; }
			nelem += tmp;
		}

		xbuf = (char *)NCI_Malloc (nelem * varp->esize);
		err	 = nczipioiconvert (cbuf, xbuf, buftype, varp->etype, nelem);
		if (err != NC_NOERR) return err;
	} else {
		xbuf = cbuf;
	}

	err = nczipioi_iput_varn (nczipp, varid, num, starts, counts, xbuf, buf, reqid);
	if (err != NC_NOERR) return err;
	if (reqid != NULL) { (*reqid) *= 2; }

	if (cbuf != buf && cbuf != xbuf) NCI_Free (cbuf);

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_IPUT)

	return NC_NOERR;
}

int nczipio_bput_varn (void *ncdp,
					   int varid,
					   int num,
					   MPI_Offset *const *starts,
					   MPI_Offset *const *counts,
					   const void *buf,
					   MPI_Offset bufcount,
					   MPI_Datatype buftype,
					   int *reqid,
					   int reqMode) {
	int err;
	int i, j;
	void *cbuf = (void *)buf;
	void *xbuf = (void *)buf;
	MPI_Offset nelem, tmp;
	NC_zip_var *varp;
	NC_zip *nczipp = (NC_zip *)ncdp;

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_START (NC_ZIP_TIMER_IPUT)

	if (reqMode == NC_REQ_INDEP) { DEBUG_RETURN_ERROR (NC_ENOTSUPPORT); }

	if (varid < 0 || varid >= nczipp->vars.cnt) { DEBUG_RETURN_ERROR (NC_EINVAL); }
	varp = nczipp->vars.data + varid;

	if (varp->isrec) {
		for (i = 0; i < num; i++) {
			if (nczipp->recsize < starts[i][0] + counts[i][0]) {
				nczipp->recsize = starts[i][0] + counts[i][0];
			}
		}
	}

	if (varp->varkind == NC_ZIP_VAR_RAW) {
		err = nczipp->driver->bput_varn (nczipp->ncp, varp->varid, num, starts, counts, buf,
										 bufcount, buftype, reqid, reqMode);
		if (err != NC_NOERR) { return err; }
		if (reqid != NULL) { *reqid = *reqid * 2 + 1; }
		return NC_NOERR;
	}

	nelem = 0;
	for (i = 0; i < num; i++) {
		tmp = 1;
		for (j = 0; j < varp->ndim; j++) { tmp *= counts[i][j]; }
		nelem += tmp;
	}
	xbuf = (char *)NCI_Malloc (nelem * varp->esize);

	if (buftype != varp->etype) {
		err = nczipioiconvert (cbuf, xbuf, buftype, varp->etype, nelem);
		if (err != NC_NOERR) return err;
	} else {
		memcpy (xbuf, cbuf, nelem * varp->esize);
	}

	err = nczipioi_iput_varn (nczipp, varid, num, starts, counts, xbuf, buf, reqid);
	if (err != NC_NOERR) return err;
	if (reqid != NULL) { (*reqid) *= 2; }

	if (cbuf != buf && cbuf != xbuf) NCI_Free (cbuf);

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_TOTAL)

	NC_ZIP_TIMER_STOP (NC_ZIP_TIMER_IPUT)

	return NC_NOERR;
}

int nczipio_get_vard (void *ncdp,
					  int varid,
					  MPI_Datatype filetype,
					  void *buf,
					  MPI_Offset bufcount,
					  MPI_Datatype buftype,
					  int reqMode) {
	int err;
	NC_zip *nczipp = (NC_zip *)ncdp;

	DEBUG_RETURN_ERROR (NC_ENOTSUPPORT);

	err = nczipp->driver->get_vard (nczipp->ncp, varid, filetype, buf, bufcount, buftype, reqMode);
	if (err != NC_NOERR) return err;

	return NC_NOERR;
}

int nczipio_put_vard (void *ncdp,
					  int varid,
					  MPI_Datatype filetype,
					  const void *buf,
					  MPI_Offset bufcount,
					  MPI_Datatype buftype,
					  int reqMode) {
	int err;
	NC_zip *nczipp = (NC_zip *)ncdp;

	DEBUG_RETURN_ERROR (NC_ENOTSUPPORT);

	err = nczipp->driver->put_vard (nczipp->ncp, varid, filetype, buf, bufcount, buftype, reqMode);
	if (err != NC_NOERR) return err;

	return NC_NOERR;
}
