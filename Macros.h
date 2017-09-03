#ifndef _MACROS_H_
#define _MACROS_H_

#define PLOT_RESOLUTION 101

#define IJ(i, j) ((i) + GRID_SIZE*(j))

#if SINGLE_PRECISION
#define Real float
#else
#define Real double
#endif

#endif
