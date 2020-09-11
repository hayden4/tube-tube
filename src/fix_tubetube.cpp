#include "fix_tubetube.h"

#include <mpi.h>
#include <cstring>
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "modify.h"
#include "input.h"
#include "variable.h"
#include "error.h"
#include "force.h"
#include "comm.h"
#include "memory.h"

#include <cstdio>

using namespace LAMMPS_NS;
using namespace FixConst;

FixTubeTube::FixTubeTube(LAMMPS * lmp, int narg, char **arg) :
	Fix(lmp, narg, arg)
{
	atom->nbond_max = nbond_max;
	comm_forward = 5 * atom->nbond_max + 1;
	comm_reverse = 3 * atom->nbond_max;
	ghost_capacity = 0;
}

FixTubeTube::~FixTubeTube()
{
	if (ghost_capacity > 0)
	{
		memory->destroy(ghost_atom_tag);
		memory->destroy(ghost_bond_tags);
		memory->destroy(ghost_bond_types);
		memory->destroy(ghost_bond_x);
		memory->destroy(ghost_bond_f);
	}
}

int FixTubeTube::setmask()
{
	int mask = 0;
	mask |= PRE_FORCE;
	return mask;
}

void FixTubeTube::setup_pre_force(int vflag)
{
	pre_force(vflag);
}

void FixTubeTube::pre_force(int vflag)
{
	ensureCapacity(atom->nghost);

	// Clear ghost bond tags & forces
	for (int i = 0; i < atom->nghost; i++)
	{
		for (int j = 0; j < atom->nbond_max; j++)
		{
			atom->ghost_bond_tags[i][j] = 0;
			atom->ghost_bond_f[i][3*j+0] = 0;
			atom->ghost_bond_f[i][3*j+1] = 0;
			atom->ghost_bond_f[i][3*j+2] = 0;
		}
	}
	
	comm->forward_comm_fix(this);
}

void FixTubeTube::post_force(int vflag)
{
	comm->reverse_comm_fix(this);
}


int FixTubeTube::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
	int m = 0;
	if (pbc_flag == 0)
	{
		for (int i = 0; i < n; i++)
		{
			int a = atom->map(atom->tag[list[i]]);

			buf[m++] = ubuf(atom->tag[a]).d;


			if (a >= atom->nlocal)
			{
				a -= atom->nlocal;
				for (int j = 0; j < atom->nbond_max; j++)
				{
					buf[m++] = ubuf(atom->ghost_bond_tags[a][j]).d;
					buf[m++] = ubuf(atom->ghost_bond_types[a][j]).d;
					buf[m++] = atom->ghost_bond_x[a][3*j+0];
					buf[m++] = atom->ghost_bond_x[a][3*j+1];
					buf[m++] = atom->ghost_bond_x[a][3*j+2];
				}
			}
			else
			{
				for (int j = 0; j < atom->nbond_max; j++)
				{
					if (j >= atom->num_bond[a])
					{
						buf[m] = (tagint) ubuf(0).d;
						m += 5;
					}
					else
					{
						int b = atom->map(atom->bond_atom[a][j]);

						buf[m++] = ubuf(atom->tag[b]).d;
						buf[m++] = ubuf(atom->bond_type[a][j]).d;
						buf[m++] = atom->x[b][0];
						buf[m++] = atom->x[b][1];
						buf[m++] = atom->x[b][2];
					}
				}
			}
		}
	}
	else
	{
		double dx, dy, dz;

		if (domain->triclinic == 0)
		{
			dx = pbc[0] * domain->xprd;
			dy = pbc[1] * domain->yprd;
			dz = pbc[2] * domain->zprd;
		}
		else
		{
			dx = pbc[0];
			dy = pbc[1];
			dz = pbc[2];
		}

		for (int i = 0; i < n; i++)
		{
			int a = atom->map(atom->tag[list[i]]);

			buf[m++] = ubuf(atom->tag[a]).d;

			if (a >= atom->nlocal)
			{
				a -= atom->nlocal;
				for (int j = 0; j < atom->nbond_max; j++)
				{
					buf[m++] = ubuf(atom->ghost_bond_tags[a][j]).d;
					buf[m++] = ubuf(atom->ghost_bond_types[a][j]).d;
					buf[m++] = atom->ghost_bond_x[a][3*j+0];
					buf[m++] = atom->ghost_bond_x[a][3*j+1];
					buf[m++] = atom->ghost_bond_x[a][3*j+2];
				}
			}
			else
			{
				for (int j = 0; j < atom->nbond_max; j++)
				{
					if (j >= atom->num_bond[a])
					{
						buf[m] = (tagint) ubuf(0).d;
						m += 5;
					}
					else
					{
						int b = atom->map(atom->bond_atom[a][j]);

						buf[m++] = ubuf(atom->tag[b]).d;
						buf[m++] = ubuf(atom->bond_type[a][j]).d;
						buf[m++] = atom->x[b][0];
						buf[m++] = atom->x[b][1];
						buf[m++] = atom->x[b][2];
					}
				}
			}
		}
	}

	return m;
}

void FixTubeTube::unpack_forward_comm(int n, int first, double *buf)
{
	int m = 0;
	for (int k = 0; k < n; k++)
	{
		int i = k + first - atom->nlocal;

		atom->ghost_atom_tag[i] = (tagint) ubuf(buf[m++]).i;

		for (int j = 0; j < atom->nbond_max; j++)
		{
			tagint b = (tagint) ubuf(buf[m++]).i;

			if (b == 0)
			{
				//atom->ghost_bond_tags[i][j] = 0;
				m += 4;
			}
			else
			{
				atom->ghost_bond_tags[i][j] = b;
				atom->ghost_bond_types[i][j] = (int) ubuf(buf[m++]).i;
				atom->ghost_bond_x[i][3*j+0] = buf[m++];
				atom->ghost_bond_x[i][3*j+1] = buf[m++];
				atom->ghost_bond_x[i][3*j+2] = buf[m++];
			}
		}
	}
}

int FixTubeTube::pack_reverse_comm(int n, int first, double *buf)
{
	int m = 0;

	for (int k = 0; k < n; k++)
	{
		int i = k + first - atom->nlocal;

		for (int j = 0; j < atom->nbond_max; j++)
		{
			buf[m++] = atom->ghost_bond_f[i][3*j+0];
			buf[m++] = atom->ghost_bond_f[i][3*j+1];
			buf[m++] = atom->ghost_bond_f[i][3*j+2];
		}

	}

	return m;
}

void FixTubeTube::unpack_reverse_comm(int n, int *list, double *buf)
{
	int m = 0;
	// Iterate over the list
	for (int i = 0; i < n; i++)
	{
		// Get local atom index
		int a = atom->map(atom->tag[list[i]]);

		// Check if it's a local atom
		if (a >= atom->nlocal)
		{
			// If not, unpack into ghost force
			a -= atom->nlocal;
			for (int j = 0; j < atom->nbond_max; j++)
			{
				atom->ghost_bond_f[a][3*j+0] += buf[m++];
				atom->ghost_bond_f[a][3*j+1] += buf[m++];
				atom->ghost_bond_f[a][3*j+2] += buf[m++];
			}
		}
		else
		{
			// If so, unpack into atom force
			for (int j = 0; j < atom->nbond_max; j++)
			{
				if (j >= atom->num_bond[a])
				{
					m += 3;
				}
				else
				{
					int b = atom->map(atom->bond_atom[a][j]);

					atom->f[b][0] += buf[m++];
					atom->f[b][1] += buf[m++];
					atom->f[b][2] += buf[m++];
				}
			}
		}
	}
}

void FixTubeTube::ensureCapacity(int n)
{
	// Only allocate valid capacities
	if (n < 1) return;

	// Already have capacity
	if (n <= ghost_capacity) return;

	// Grow arrays
	ghost_atom_tag = memory->grow(atom->ghost_atom_tag, n, "atom:ghost_atom_tag");
	ghost_bond_tags = memory->grow(atom->ghost_bond_tags, n, nbond_max, "atom:ghost_bond_tags");
	ghost_bond_types = memory->grow(atom->ghost_bond_types, n, nbond_max, "atom:ghost_bond_types");
	ghost_bond_x = memory->grow(atom->ghost_bond_x, n, nbond_max*3, "atom:ghost_bond_x");
	ghost_bond_f = memory->grow(atom->ghost_bond_f, n, nbond_max*3, "atom:ghost_bond_f");

	ghost_capacity = n;
}

double FixTubeTube::memory_usage()
{
	double bytes = 0;

	bytes += ghost_capacity * sizeof(tagint);
	bytes += ghost_capacity * nbond_max * sizeof(tagint);
	bytes += ghost_capacity * nbond_max * sizeof(int);
	bytes += ghost_capacity * nbond_max * 3 * sizeof(double);
	bytes += ghost_capacity * nbond_max * 3 * sizeof(double);

	return bytes;
}
