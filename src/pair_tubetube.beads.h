#ifdef PAIR_CLASS

PairStyle(tubetube,PairTubeTube)

#else

#ifndef LMP_PAIR_TUBE_TUBE_H
#define LMP_PAIR_TUBE_TUBE_H

#include "pair.h"

// ------ DEBUG ------
#include <vector>
// -------------------

namespace LAMMPS_NS {

class PairTubeTube : public Pair {
 public:
  PairTubeTube(class LAMMPS *);
  virtual ~PairTubeTube();

  virtual void compute(int, int);
  void settings(int, char**);
  void coeff(int, char**);

  void write_data(FILE *fp);
  void write_data_all(FILE *fp);

  double init_one(int i, int j);

 protected:
  double cut_global;
  double **cut;
  double **hamaker;	// Hamaker constant A
  double **radius;	// Tube radius a
  double **vdw;		// vdW factor (determined by init_one)
  double **sigma;
  double **sigma6;

  // Check if the neighbor list contains the specified atom
  bool nl_contains(tagint, int*, int);

  void calculate_force(
  		double* ix, double* jx, double* kx, double* lx,
		int type1, int type2, int atom_i, int atom_j,
		double* fi, double* fj, double* fk, double* fl,
		int);

  void allocate();
};

}

#endif
#endif
