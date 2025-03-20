/****************************************************************************/
/* TREECODE.H: define various things for treecode.c and treeio.c.           */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#ifndef _treecode_h
#define _treecode_h

#include <mpi.h>
#include <stdint.h>
#include "treedefs.h"
#include <cstddef>

typedef struct {
 uint32_t index;
 real phi;
 vector acc;
} body_update_t;

/*
 * Parameters, state variables, and diagnostics for N-body integration.
 */

global string infile;                   /* file name for snapshot input     */

global string outfile;                  /* file name for snapshot output    */

global string savefile;                 /* file name for state output       */

#if defined(USEFREQ)

global real freq;                       /* basic integration frequency      */

global real freqout;                    /* data output frequency            */

#else

global real dtime;                      /* basic integration timestep       */

global real dtout;                      /* data output timestep             */

#endif

global real tstop;                      /* time to stop calculation         */

global string headline;                 /* message describing calculation   */

global real tnow;                       /* current value of time            */

global real tout;                       /* time of next output              */

global int nstep;                       /* number of time-steps             */

global int nbody;                       /* number of bodies in system       */

global bodyptr bodytab;                 /* points to array of bodies        */

global cellptr celltab;

global body_update_t* local_body_updates_buffer;

global body_update_t* all_body_updates_buffer;



global int mpi_rank; 

global int mpi_numproc;

global MPI_Datatype mpi_body_update_type;

global int mpi_depth;
global int omp_threshold;
global int cuda_blocksize;

/*
 * Prototypes for I/O routines.
 */

void inputdata(void);                   /* read initial data file           */
void startoutput(void);                 /* open files for output            */
void forcereport(void);                 /* report on force calculation      */
void output(void);                      /* perform output operation         */
void savestate(string);                 /* save system state                */
void restorestate(string);              /* restore system state             */

#endif /* ! _treecode_h */
