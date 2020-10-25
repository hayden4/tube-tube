#ifdef PAIR_CLASS

PairStyle(tubetube,PairTubeTube)

#else

#ifndef LMP_PAIR_TUBE_TUBE_H
#define LMP_PAIR_TUBE_TUBE_H

#include "pair.h"

namespace LAMMPS_NS {

/**
 * The Pair Tube-Tube class implements a pairwise interaction between bonds.
 */
class PairTubeTube : public Pair {
 public:

  // Constructor & Destructor
  PairTubeTube(class LAMMPS *);
  virtual ~PairTubeTube();

  // Compute method
  virtual void compute(int, int);

  // Set interaction parameters
  void settings(int, char**);
  void coeff(int, char**);
  double init_one(int i, int j);

  // Read and write parameters
  void write_data(FILE *fp);
  void write_data_all(FILE *fp);
  void write_restart(FILE *fp);
  void write_restart_settings(FILE *fp);
  void read_restart(FILE *fp);
  void read_restart_settings(FILE *fp);

 protected:
  // Cut variables
  double cut_global;
  double **cut;
  
  // Allocate memory 
  void allocate();


  // Potential variables
  double **hamaker;	// Hamaker constant A
  double **radius;	// Tube radius a
  double **xi;		// Repulsion distance xi
  double **vdw;		// vdW factor (determined by init_one)

  // Repulsion settings
  int repulsion_type;
  static const int NO_REPULSION = 0;
  static const int POWER_REPULSION = 1;
  static const int EXP_REPULSION = 2;


  // Double checking method
  bool nl_contains(tagint, int*, int);

  // Force calculation method
  void calculate_force(
  		double* ix, double* jx, double* kx, double* lx,
		int type1, int type2, int atom_i, int atom_j,
		double* fi, double* fj, double* fk, double* fl);
};

}

#endif
#endif
