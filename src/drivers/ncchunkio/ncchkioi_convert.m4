dnl Process this m4 file to produce 'C' language file.
dnl
dnl If you see this line, you can ignore the next one.
/* Do not edit this file. It is produced from the corresponding .m4 source */
dnl
/*
 *  Copyright (C) 2018, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */
dnl
include(`foreach.m4')dnl
include(`utils.m4')dnl
dnl
define(`upcase', `translit(`$*', `a-z', `A-Z')')dnl
dnl
define(`SWOUT',dnl
`dnl
        if (outtype == $1) {
            for(i = 0; i < N; i++){
                (($2*)outbuf)[i] = ($2)((($3*)inbuf)[i]);
            }
            return NC_NOERR;
        }
')dnl
dnl
define(`SWIN',dnl
`dnl
    if (intype == $1){
        
foreach(`dt', (`(`MPI_BYTE', `char')', dnl
    `(`MPI_CHAR', `char')', dnl
    `(`MPI_SIGNED_CHAR', `signed char')', dnl
    `(`MPI_UNSIGNED_CHAR', `unsigned char')', dnl
    `(`MPI_SHORT', `short')', dnl
    `(`MPI_UNSIGNED_SHORT', `unsigned short')', dnl
    `(`MPI_INT', `int')', dnl
    `(`MPI_UNSIGNED', `unsigned int')', dnl
    `(`MPI_FLOAT', `float')', dnl
    `(`MPI_DOUBLE', `double')', dnl
    `(`MPI_LONG_LONG_INT', `long long')', dnl
    `(`MPI_UNSIGNED_LONG_LONG', `unsigned long long')', dnl
    ), `SWOUT(translit(dt, `()'), $2)')dnl
        DEBUG_RETURN_ERROR(NC_EBADTYPE);;
    }
')dnl
dnl
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <pnc_debug.h>
#include <common.h>
#include <ncchkio_driver.h>

int ncchkioiconvert(void *inbuf, void *outbuf, MPI_Datatype intype, MPI_Datatype outtype, int N) {
    int i;

foreach(`dt', (`(`MPI_BYTE', `char')', dnl
        `(`MPI_CHAR', `char')', dnl
        `(`MPI_SIGNED_CHAR', `signed char')', dnl
        `(`MPI_UNSIGNED_CHAR', `unsigned char')', dnl
        `(`MPI_SHORT', `short')', dnl
        `(`MPI_UNSIGNED_SHORT', `unsigned short')', dnl
        `(`MPI_INT', `int')', dnl
        `(`MPI_UNSIGNED', `unsigned int')', dnl
        `(`MPI_FLOAT', `float')', dnl
        `(`MPI_DOUBLE', `double')', dnl
        `(`MPI_LONG_LONG_INT', `long long')', dnl
        `(`MPI_UNSIGNED_LONG_LONG', `unsigned long long')', dnl
        ), `SWIN(translit(dt, `()'))')dnl
    DEBUG_RETURN_ERROR(NC_EBADTYPE);;

    return NC_NOERR;
}