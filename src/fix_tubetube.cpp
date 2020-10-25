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

/**
 * FixTubeTube Constructor
 *
 * Sets local variables necessary for communication.
 */
FixTubeTube::FixTubeTube(LAMMPS * lmp, int narg, char **arg) :
	Fix(lmp, narg, arg)
{
	atom->nbond_max = nbond_max;
	comm_forward = 4 * atom->nbond_max;
	comm_reverse = 3 * atom->nbond_max;
	ghost_capacity = 0;
}

/**
 * FixTubeTube Destructor
 *
 * Destroys any allocated memory.
 */
FixTubeTube::~FixTubeTube()
{
	if (ghost_capacity > 0)
	{
		memory->destroy(ghost_bond_tags);
		memory->destroy(ghost_bond_x);
		memory->destroy(ghost_bond_f);
	}
}

/**
 * Sets the fix mask which determines which fix methods are called.
 * PRE_FORCE and POST_FORCE are set.
 */
int FixTubeTube::setmask()
{
	int mask = 0;
	mask |= PRE_FORCE;
	mask |= POST_FORCE;
	return mask;
}

/**
 * Pre-force call during integration setup. Forward the call to
 * the normal pre-force method.
 */
void FixTubeTube::setup_pre_force(int vflag)
{
	pre_force(vflag);
}

/**
 * Pre-force call during integration.
 */
void FixTubeTube::pre_force(int vflag)
{
	// Ensure there's memory available to store ghost bonded atoms
	ensureCapacity(atom->nghost);

	// Clear ghost bond tags & forces
	// Other arrays will be overwritten as necessary from forward
	// communication. 
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
	
	// Perform forward communication of ghost bonded atoms
	comm->forward_comm_fix(this);
}

/**
 * Post-force call during integration.
 */
void FixTubeTube::post_force(int vflag)
{
	// Perform reverse communication of ghost bonded atoms
	comm->reverse_comm_fix(this);
}

/**
 * Pack atoms for forward communication
 * Atom tags and positions need to be communicated
 */
int FixTubeTube::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
	int m = 0;

	// Determine if periodic boundary conditions need to be taken into account
	if (pbc_flag == 0)
	{
		// Iterate over list of ghost atoms
		for (int i = 0; i < n; i++)
		{
			// Get local id of atom to transfer
			int a = atom->map(atom->tag[list[i]]);

			// Add a's tag to the buffer
			buf[m++] = ubuf(atom->tag[a]).d;

			if (a >= atom->nlocal)
			{
				// If a is not local, forward its ghost bonded atoms

				// Get ghost bonded atom index
				a -= atom->nlocal;

				// Add ghost bonded tags and positions to the buffer
				for (int j = 0; j < atom->nbond_max; j++)
				{
					buf[m++] = ubuf(atom->ghost_bond_tags[a][j]).d;
					buf[m++] = atom->ghost_bond_x[a][3*j+0];
					buf[m++] = atom->ghost_bond_x[a][3*j+1];
					buf[m++] = atom->ghost_bond_x[a][3*j+2];
				}
			}
			else
			{
				// If a is local, add its bonded atoms to the buffer
				for (int j = 0; j < atom->nbond_max; j++)
				{
					// If a has less than nbond_max bonded atoms, set the
					// tag to 0 and skip
					if (j >= atom->num_bond[a])
					{
						buf[m] = (tagint) ubuf(0).d;
						m += 4;
					}
					else
					{
						int b = atom->map(atom->bond_atom[a][j]);

						buf[m++] = ubuf(atom->tag[b]).d;
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
		// If periodic boundary conditions are on, compute the deltas and add them
		// to positions. Algorithm is otherwise the same as the non-periodic code
		// above.
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
						m += 4;
					}
					else
					{
						int b = atom->map(atom->bond_atom[a][j]);

						buf[m++] = ubuf(atom->tag[b]).d;
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

/**
 * Unpacks buffer recieved from forward communication into local arrays.
 */
void FixTubeTube::unpack_forward_comm(int n, int first, double *buf)
{
	int m = 0;

	// Iterate over the buffer
	for (int k = 0; k < n; k++)
	{
		// Get ghost bonded atom index
		int i = k + first - atom->nlocal;

		// Unpack ghost bonded atoms
		for (int j = 0; j < atom->nbond_max; j++)
		{
			// Get ghost bonded atom tag
			tagint b = (tagint) ubuf(buf[m++]).i;

			// If tag is 0, skip
			if (b == 0)
			{
				m += 4;
			}
			else
			{
				atom->ghost_bond_tags[i][j] = b;
				atom->ghost_bond_x[i][3*j+0] = buf[m++];
				atom->ghost_bond_x[i][3*j+1] = buf[m++];
				atom->ghost_bond_x[i][3*j+2] = buf[m++];
			}
		}
	}
}

/**
 * Packs a buffer to reverse communication after a force step.
 * Only forces need to be packed.
 */
int FixTubeTube::pack_reverse_comm(int n, int first, double *buf)
{
	int m = 0;

	// Iterate over ghost bonded atom arrays
	for (int k = 0; k < n; k++)
	{
		// Get ghost bonded atom index
		int i = k + first - atom->nlocal;

		// Pack forces
		for (int j = 0; j < atom->nbond_max; j++)
		{
			buf[m++] = atom->ghost_bond_f[i][3*j+0];
			buf[m++] = atom->ghost_bond_f[i][3*j+1];
			buf[m++] = atom->ghost_bond_f[i][3*j+2];
		}

	}

	return m;
}

/**
 * Unpack the reverse communicated buffer into local arrays.
 */
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
				// If we're past the number of bonds a has, then
				// skip the remaining forces
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

/**
 * Grow arrays to ensure there's enough memory for ghost bonded atoms transfered.
 */
void FixTubeTube::ensureCapacity(int n)
{
	// Only allocate valid capacities
	if (n < 1) return;

	// Already have capacity
	if (n <= ghost_capacity) return;

	// Grow arrays
	ghost_bond_tags = memory->grow(atom->ghost_bond_tags, n, nbond_max, "atom:ghost_bond_tags");
	ghost_bond_x = memory->grow(atom->ghost_bond_x, n, nbond_max*3, "atom:ghost_bond_x");
	ghost_bond_f = memory->grow(atom->ghost_bond_f, n, nbond_max*3, "atom:ghost_bond_f");

	// Set the new capacity
	ghost_capacity = n;
}

/**
 * Return an estimation of the memory usage of fix_tubetube
 */
double FixTubeTube::memory_usage()
{
	double bytes = 0;

	bytes += ghost_capacity * sizeof(tagint);
	bytes += ghost_capacity * nbond_max * sizeof(tagint);
	bytes += ghost_capacity * nbond_max * 3 * sizeof(double);
	bytes += ghost_capacity * nbond_max * 3 * sizeof(double);

	return bytes;
}
