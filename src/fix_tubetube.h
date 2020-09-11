#ifdef FIX_CLASS

FixStyle(tubetube,FixTubeTube)

#else

#ifndef LMP_FIX_TUBETUBE_H
#define LMP_FIX_TUBETUBE_H

#include "fix.h"

namespace LAMMPS_NS {

	class FixTubeTube : public Fix {
	public:
		FixTubeTube(class LAMMPS *, int, char **);
		virtual ~FixTubeTube();

		int setmask();

		virtual void setup_pre_force(int);
		virtual void pre_force(int);
		virtual void post_force(int);

		int pack_forward_comm(int, int *, double *, int, int *);
		int pack_reverse_comm(int, int, double*);
		void unpack_forward_comm(int, int, double *);
		void unpack_reverse_comm(int, int*, double*);

		double memory_usage();
	private:
		const static int nbond_max = 4;	// maximum number of bonds to be transmitted during communication

		tagint 	 *ghost_atom_tag;	// ghost atom's tag
		tagint 	**ghost_bond_tags;	// tags of atoms in the ghost atom's bonds 
		int 	**ghost_bond_types;	// types of ghost atom's bonds
		double	**ghost_bond_x;		// positions of atoms in the ghost atom's bonds
		double	**ghost_bond_f;

		int ghost_capacity;		// ghost array capacity
		void ensureCapacity(int);	// ensures ghost arrays have specified capacity
	};
}

#endif
#endif
