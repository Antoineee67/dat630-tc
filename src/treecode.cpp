/****************************************************************************/
/* TREECODE.C: new hierarchical N-body code.                                */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#include "stdinc.h"
#include "mathfns.h"
#include "vectmath.h"
#include "getparam.h"
#define global                                  /* don't default to extern  */
#include "treecode.h"

#include <assert.h>
//#include <cuda_runtime_api.h>

#include "treedefs.h"
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>

#include "omp.h"
#include "mpi.h"
#include "cudagravsum.cuh"


/*
 * Default values for input parameters.
 */

string defv[] = {
    ";Hierarchical N-body code "
#if defined(QUICKSCAN)
                                    "(quick scan)",
#else
    "(theta scan)",
#endif
    "in=", ";Input file with initial conditions",
    "out=", ";Output file of N-body frames",
#if defined(USEFREQ)
    "freq=32.0",                ";Leapfrog integration frequency",
#else
    "dtime=1/32", ";Leapfrog integration timestep",
#endif
    "eps=0.025", ";Density smoothing length",
#if !defined(QUICKSCAN)
    "theta=1.0", ";Force accuracy parameter",
#endif
    "usequad=false", ";If true, use quad moments",
    "options=", ";Various control options",
    "tstop=2.0", ";Time to stop integration",
#if defined(USEFREQ)
    "freqout=4.0",              ";Data output frequency",
#else
    "dtout=1/4", ";Data output timestep",
#endif
    "nbody=4096", ";Number of bodies for test run",
    "seed=123", ";Random number seed for test run",
    "save=", ";Write state file as code runs",
    "restore=", ";Continue run from state file",
    "VERSION=1.4", ";Joshua Barnes  February 21 2001",
    "mpi_depth=5", ";MPI threshold for which tree depth to filter by MPI node",
    "omp_threshold=6",
    ";OMP threshold to which tree depth to parallelize. Below this depth new threads are not created",
    NULL,
};

/* Prototypes for local procedures. */

local void treeforce(void); /* do force calculation     */
local void stepsystem(void); /* advance by one time-step */
local void startrun(void); /* initialize system state  */
local void testdata(void); /* generate test data       */
local void dataExchange(); /* exchange data between processes */

void create_mpi_body_update_type(MPI_Datatype *mpi_body_update_type) {
    // Define the lengths of each block
    int lengths[3] = {1, 1, NDIM};

    // Define the displacements of each block
    MPI_Aint displacements[3];
    displacements[0] = offsetof(body_update_t, index);
    displacements[1] = offsetof(body_update_t, phi);
    displacements[2] = offsetof(body_update_t, acc);

    // Define the types of each block

    MPI_Datatype types[3] = {MPI_UINT32_T, mpi_real_type, mpi_real_type};

    // Create the MPI data type
    MPI_Type_create_struct(3, lengths, displacements, types, mpi_body_update_type);
    MPI_Type_commit(mpi_body_update_type);
}

/*
 * MAIN: toplevel routine for hierarchical N-body code.
 */

int main(int argc, string argv[]) {
    //sleep(20);


    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    create_mpi_body_update_type(&mpi_body_update_type);

    int deviceCount;
    //cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }


    double start_time;
    initparam(argv, defv); /* initialize param access  */
    headline = defv[0] + 1; /* skip ";" in headline     */
    startrun(); /* get params & input data  */

    if (mpi_depth < 2 && mpi_numproc > 1) {
        error("mpi_depth must be greater than 1 when using more than 1 MPI process\n");
    }
    if (mpi_numproc < 2) {
        // If we have only one process, we don't need to run MPI depth filtering
        mpi_depth = 0;
    }
    if (mpi_rank == 0) {
        start_time = omp_get_wtime();
        startoutput();
    } /* activate output code     */

    cuda_gravsum_init();
    if (nstep == 0) {
        /* if data just initialized */
        treeforce(); /* do complete calculation  */
        dataExchange();
        if (mpi_rank == 0)
            output(); /* and report diagnostics   */
    }
#if defined(USEFREQ)
    if (freq != 0.0)                            /* if time steps requested  */
        while (tstop - tnow > 0.01/freq) {      /* while not past tstop     */
            stepsystem();                       /* advance step by step     */
            output();                           /* and output results       */
        }
#else
    if (dtime != 0.0) /* if time steps requested  */
        while (tstop - tnow > 0.01 * dtime) {
            /* while not past tstop     */
            stepsystem(); /* advance step by step     */
            if (mpi_rank == 0)
                output(); /* and output results       */
        }
#endif
    if (mpi_rank == 0) {
        double stop_time = omp_get_wtime();
        printf("Time elapsed: %g sec\n", stop_time - start_time);
    }
    MPI_Finalize();
    return (0); /* end with proper status   */
}

/*
 * TREEFORCE: common parts of force calculation.
 */

local void treeforce(void) {
    bodyptr p;

    for (p = bodytab; p < bodytab + nbody; p++) /* loop over all bodies     */
        Update(p) = TRUE; /* update all forces        */
    maketree(bodytab, nbody); /* construct tree structure */
    cuda_update_body_cell_data();
    gravcalc(); /* compute initial forces   */
    cuda_gravsum_dispatch();
    if (mpi_rank == 0)
        forcereport(); /* print force statistics   */
}

/*
 * STEPSYSTEM: advance N-body system using simple leap-frog.
 */

local void stepsystem(void) {
#if defined(USEFREQ)
    real dtime = 1.0 / freq;                    /* set basic time-step      */
#endif
    bodyptr p;


    for (p = bodytab; p < bodytab + nbody; p++) {
        /* loop over all bodies     */
        ADDMULVS(Vel(p), Acc(p), 0.5 * dtime); /* advance v by 1/2 step    */
        ADDMULVS(Pos(p), Vel(p), dtime); /* advance r by 1 step      */
    }
    treeforce(); /* perform force calc.      */
    dataExchange();
    for (p = bodytab; p < bodytab + nbody; p++) {
        /* loop over all bodies     */
        ADDMULVS(Vel(p), Acc(p), 0.5 * dtime); /* advance v by 1/2 step    */
    }
    nstep++; /* count another time step  */
    tnow = tnow + dtime; /* finally, advance time    */
}

/*
 * STARTRUN: startup hierarchical N-body code.
 */

local void startrun(void) {
    mpi_depth = getiparam("mpi_depth");
    omp_threshold = getiparam("omp_threshold");
#if !defined(USEFREQ)
    double dt1, dt2;
#endif

    infile = getparam("in"); /* set I/O file names       */
    outfile = getparam("out");
    savefile = getparam("save");
    if (strnull(getparam("restore"))) {
        /* if starting a new run    */
        eps = getdparam("eps"); /* get input parameters     */
        eps2 = eps * eps; /* and square of eps        */
#if defined(USEFREQ)
        freq = getdparam("freq");
#else
        dtime = (sscanf(getparam("dtime"), "%lf/%lf", &dt1, &dt2) == 2 ? dt1 / dt2 : getdparam("dtime"));
#endif
#if !defined(QUICKSCAN)
        theta = getdparam("theta");
#endif
        usequad = getbparam("usequad");
        tstop = getdparam("tstop");
#if defined(USEFREQ)
        freqout = getdparam("freqout");
#else
        dtout = (sscanf(getparam("dtout"), "%lf/%lf", &dt1, &dt2) == 2 ? dt1 / dt2 : getdparam("dtout"));
#endif
        options = getparam("options");
        if (!strnull(infile)) /* if data file was given   */
            inputdata(); /* then read inital data    */
        else {
            /* else make initial data   */
            nbody = getiparam("nbody"); /* get number of bodies     */
            if (nbody < 1) /* check for silly values   */
                error("startrun: absurd value for nbody\n");
            srandom(getiparam("seed")); /* set random number gen.   */
            testdata(); /* and make plummer model   */
            tnow = 0.0; /* reset elapsed model time */
        }
        rsize = 1.0; /* start root w/ unit cube  */
        nstep = 0; /* begin counting steps     */
        tout = tnow; /* schedule first output    */
    } else {
        /* else restart old run     */
        restorestate(getparam("restore")); /* read in state file       */
        if (getparamstat("eps") & ARGPARAM) /* if given, set new params */
            {
                eps = getdparam("eps");
                eps2 = eps * eps;
            }
#if !defined(QUICKSCAN)
        if (getparamstat("theta") & ARGPARAM)
            theta = getdparam("theta");
#endif
        if (getparamstat("usequad") & ARGPARAM)
            usequad = getbparam("usequad");
        if (getparamstat("options") & ARGPARAM)
            options = getparam("options");
        if (getparamstat("tstop") & ARGPARAM)
            tstop = getdparam("tstop");
#if defined(USEFREQ)
        if (getparamstat("freqout") & ARGPARAM)
            freqout = getdparam("freqout");
        if (scanopt(options, "new-tout"))       /* if output time reset     */
            tout = tnow + 1 / freqout;          /* then offset from now     */
#else
        dtout = (sscanf(getparam("dtout"), "%lf/%lf", &dt1, &dt2) == 2 ? dt1 / dt2 : getdparam("dtout"));
        if (scanopt(options, "new-tout")) /* if output time reset     */
            tout = tnow + dtout; /* then offset from now     */
#endif
    }
}

/*
 * TESTDATA: generate Plummer model initial conditions for test runs,
 * scaled to units such that M = -4E = G = 1 (Henon, Hegge, etc).
 * See Aarseth, SJ, Henon, M, & Wielen, R (1974) Astr & Ap, 37, 183.
 */

#define MFRAC  0.999                            /* cut off 1-MFRAC of mass  */

local void testdata(void) {
    real rsc, vsc, r, v, x, y;
    vector rcm, vcm;
    bodyptr p;

    bodytab = (bodyptr) allocate(nbody * sizeof(body));
    celltab = (cellptr) allocate(nbody * sizeof(cell));
    local_body_updates_buffer = (body_update_t *) allocate(nbody * sizeof(body_update_t));
    all_body_updates_buffer = (body_update_t *) allocate(nbody * sizeof(body_update_t));
    /* alloc space for bodies   */
    rsc = (3 * PI) / 16; /* and length scale factor  */
    vsc = rsqrt(1.0 / rsc); /* find speed scale factor  */
    CLRV(rcm); /* zero out cm position     */
    CLRV(vcm); /* zero out cm velocity     */
    for (p = bodytab; p < bodytab + nbody; p++) {
        /* loop over bodies         */
        Type(p) = BODY; /* tag as a body            */
        Mass(p) = 1.0 / nbody; /* set masses equal         */
        x = xrandom(0.0, MFRAC); /* pick enclosed mass       */
        r = 1.0 / rsqrt(rpow(x, -2.0 / 3.0) - 1); /* find enclosing radius    */
        pickshell(Pos(p), NDIM, rsc * r); /* pick position vector     */
        do {
            /* select from fn g(x)      */
            x = xrandom(0.0, 1.0); /* for x in range 0:1       */
            y = xrandom(0.0, 0.1); /* max of g(x) is 0.092     */
        } while (y > x * x * rpow(1 - x * x, 3.5)); /* using von Neumann tech   */
        v = x * rsqrt(2.0 / rsqrt(1 + r * r)); /* find resulting speed     */
        pickshell(Vel(p), NDIM, vsc * v); /* pick velocity vector     */
        ADDMULVS(rcm, Pos(p), 1.0 / nbody); /* accumulate cm position   */
        ADDMULVS(vcm, Vel(p), 1.0 / nbody); /* accumulate cm velocity   */
    }
    for (p = bodytab; p < bodytab + nbody; p++) {
        /* loop over bodies again   */
        SUBV(Pos(p), Pos(p), rcm); /* subtract cm position     */
        SUBV(Vel(p), Vel(p), vcm); /* subtract cm velocity     */
    }
}


local void dataExchange() {
    static int exchange_count = 0;
    exchange_count++;
    int32_t local_changes_count = 0;

    for (int i = 0; i < nbody; i++) {
        bodyptr body = &bodytab[i];
        if (body->updated) {
            body_update_t *update = &local_body_updates_buffer[local_changes_count];
            update->index = i;
            update->phi = body->phi;
            SETV(update->acc, body->acc);
            local_changes_count++;
        }
    }
    //Allgather receive change counts.
    int32_t receive_count[mpi_numproc];

    MPI_Allgather(&local_changes_count, 1, MPI_INT32_T, &receive_count, 1, MPI_INT32_T, MPI_COMM_WORLD);
    //AllgatherV sends all changes to all ranks.

    int32_t total_changes = 0;
    for (int i = 0; i < mpi_numproc; i++) {
        total_changes += receive_count[i];
    }

    if (total_changes != nbody && mpi_rank == 0) {
        printf("Total changes: %d, nbody: %d, exchange count %d, local changes %d, rank %d\n", total_changes, nbody,
               exchange_count, local_changes_count, mpi_rank);
    }

    body_update_t all_body_updates_buffer_local[total_changes];

    // Compute displacements
    int displs[mpi_numproc];
    displs[0] = 0;
    for (int i = 1; i < mpi_numproc; i++) {
        displs[i] = displs[i - 1] + receive_count[i - 1];
    }
    MPI_Allgatherv(local_body_updates_buffer, local_changes_count, mpi_body_update_type, all_body_updates_buffer_local,
                   (const int *) &receive_count, (const int *) &displs, mpi_body_update_type, MPI_COMM_WORLD);

    //Apply changes to all bodies.
    for (int j = 0; j < total_changes; j++) {
        body_update_t *update = &all_body_updates_buffer_local[j];
        bodyptr body = &bodytab[update->index];
        assert(update->index < nbody);
        body->phi = update->phi;
        SETV(body->acc, update->acc);
    }
}
