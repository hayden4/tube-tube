#include "pair_tubetube.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include "domain.h"
#include <vector>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairTubeTube::PairTubeTube(LAMMPS *lmp) : Pair(lmp)
{
	respa_enable = 0;
	writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairTubeTube::~PairTubeTube()
{
	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(cut);
		memory->destroy(radius);
		memory->destroy(sigma);
		memory->destroy(sigma6);
		memory->destroy(vdw);
	}
}

/* ---------------------------------------------------------------------- */

void PairTubeTube::compute(int eflag, int vflag)
{
	double** x = atom->x;

	ev_init(eflag, vflag);

	// Iterate over atoms in pair potential
	for (int ii = 0; ii < list->inum; ii++)
	{
		// Get ith atom to perform pair computations on
		int i = list->ilist[ii];
		int type1 = atom->type[i];

		// Iterate over atom i's bonded atoms
		for (int jj = 0; jj < atom->num_bond[i]; jj++)
		{
			// Get jth atom bonded to i, and map to closest image if it's a ghost atom
			int j = atom->map(atom->bond_atom[i][jj]);

			// Iterate over atom i's neighbor list
			for (int kk = 0; kk < list->numneigh[i]; kk++)
			{
				// Get kth atom in atom i's neighbor list
				int k = list->firstneigh[i][kk];
				int type2 = atom->type[k];
				k &= NEIGHMASK;

				if (nl_contains(atom->tag[k], list->ilist, list->inum))
				{
					k = atom->map(atom->tag[k]);

					for (int ll = 0; ll < atom->num_bond[k]; ll++)
					{
						int l = atom->map(atom->bond_atom[k][ll]);

						// i is local
						// k is local

						if (nl_contains(atom->tag[l], list->firstneigh[i], list->numneigh[i])) if (atom->tag[l] < atom->tag[k]) continue;
						if (nl_contains(atom->tag[i], list->firstneigh[k], list->numneigh[k])) if (atom->tag[k] < atom->tag[i]) continue;
						if (nl_contains(atom->tag[j], list->firstneigh[k], list->numneigh[k])) if (atom->tag[k] < atom->tag[i]) continue;
						if (nl_contains(atom->tag[j], list->ilist, list->inum))
						{
							if (nl_contains(atom->tag[k], list->firstneigh[j], list->numneigh[j])) if (atom->tag[j] < atom->tag[i]) continue;
							if (nl_contains(atom->tag[l], list->firstneigh[j], list->numneigh[j])) if (atom->tag[j] < atom->tag[i]) continue;
						}
						if (nl_contains(atom->tag[l], list->ilist, list->inum))
						{
							if (nl_contains(atom->tag[i], list->firstneigh[l], list->numneigh[l])) if (atom->tag[l] < atom->tag[i]) continue;
							if (nl_contains(atom->tag[j], list->firstneigh[l], list->numneigh[l])) if (atom->tag[l] < atom->tag[i]) continue;
						}

						calculate_force(atom->x[i], atom->x[j], atom->x[k], atom->x[l], type1, type2, i, k, atom->f[i], atom->f[j], atom->f[k], atom->f[l], eflag);
					}
				}
				else
				{
					if (k < atom->nlocal)
						continue;

					k = atom->map(atom->tag[k]);

					for (int ll = 0; ll < atom->nbond_max; ll++)
					{
						tagint tag_l = atom->ghost_bond_tags[k-atom->nlocal][ll];

						if (tag_l <= 0) continue;

						int l = atom->map(tag_l);

						// i is local
						// k is not local
						
						if (nl_contains(atom->tag[l], list->firstneigh[i], list->numneigh[i])) if (atom->tag[l] < atom->tag[k]) continue;
						if (nl_contains(atom->tag[j], list->ilist, list->inum))
						{
							if (nl_contains(atom->tag[k], list->firstneigh[j], list->numneigh[j])) if (atom->tag[j] < atom->tag[i]) continue;
							if (nl_contains(atom->tag[l], list->firstneigh[j], list->numneigh[j])) if (atom->tag[j] < atom->tag[i]) continue;
						}
						if (l > -1 and nl_contains(atom->tag[l], list->ilist, list->inum))
						{
							if (nl_contains(atom->tag[i], list->firstneigh[l], list->numneigh[l])) if (atom->tag[l] < atom->tag[i]) continue;
							if (nl_contains(atom->tag[j], list->firstneigh[l], list->numneigh[l])) if (atom->tag[l] < atom->tag[i]) continue;
						}

						calculate_force(atom->x[i], atom->x[j], atom->x[k], &atom->ghost_bond_x[k-atom->nlocal][3*ll], type1, type2, i, k, atom->f[i], atom->f[j], atom->f[k], &atom->ghost_bond_f[k-atom->nlocal][3*ll], eflag);
					}
				}
			}
		}
	}

	if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairTubeTube::allocate()
{
	allocated = 1;
	int n = atom->ntypes;
	
	memory->create(setflag,n+1,n+1,"pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;
	
	memory->create(cutsq,n+1,n+1,"pair:cutsq");
	memory->create(cut,n+1,n+1,"pair:cut");
	memory->create(radius,n+1,n+1,"pair:radius");
	memory->create(vdw,n+1,n+1,"pair:vdw");
	memory->create(sigma,n+1,n+1,"pair:sigma");
	memory->create(sigma6,n+1,n+1,"pair:sigma6");
}

/* ---------------------------------------------------------------------- */

void PairTubeTube::settings(int narg, char **arg)
{
	if (narg != 1) error->all(FLERR,"Illegal pair_style command");

	cut_global = force->numeric(FLERR,arg[0]);

	if (allocated) {
		for (int i = 1; i <= atom->ntypes; i++)
			for (int j = i; j <= atom->ntypes; j++)
				if (setflag[i][j]) cut[i][j] = cut_global;
	}
}

/* ---------------------------------------------------------------------- */

void PairTubeTube::coeff(int narg, char **arg)
{
	if (narg < 5 || narg > 6)
		error->all(FLERR, "Incorrect args for pair coefficients");

	if (!allocated) allocate();

	int ilo,ihi,jlo,jhi;
	force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
	force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

	double vdw_one = force->numeric(FLERR, arg[2]);
	double radius_one  = force->numeric(FLERR, arg[3]);
	double sigma_one = force->numeric(FLERR, arg[4]);

	double cut_one = cut_global;
	if (narg >= 6) cut_one = force->numeric(FLERR, arg[5]);

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {
		for (int j = MAX(jlo,i); j <= jhi; j++) {
			vdw[i][j] = vdw_one;
			radius[i][j] = radius_one;
			sigma[i][j] = sigma_one;
			cut[i][j] = cut_one;
			setflag[i][j] = 1;
			count++;
		}
	}

	if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairTubeTube::write_data(FILE *fp)
{
	fprintf(fp, "Proc %d\n", comm->me);
	fprintf(fp, "THIS IS THE TUBE TUBE POTENTIAL\n");
}

void PairTubeTube::write_data_all(FILE *fp)
{
	fprintf(fp, "THIS IS THE TUBE TUBE POTENTIAL ALL\n");
}

double PairTubeTube::init_one(int i, int j)
{
	sigma6[i][j] = pow(sigma[i][j], 6);
	sigma6[j][i] = sigma6[i][j];

	return cut[i][j];
}





bool PairTubeTube::nl_contains(tagint a, int* ls, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (atom->tag[ls[i]] == a)
		{
			return true;
		}
	}
	return false;
}

// Vector operations

double dot(double* a, double* b)
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

double norm(double* a)
{
	return sqrt(dot(a,a));
}

void cross(double* a, double* b, double* c)
{
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
}

void scale(double a, double* v)
{
	for (int i = 0; i < 3; i++)
		v[i] *= a;
}

double powint(double a, int b)
{
	double x = a;
	for (int i = 1; i < abs(b); i++)
		x *= a;
	if (b < 0)
		return 1/x;
	else
		return x;
}

void PairTubeTube::calculate_force(
	double* ix, double* jx, double* kx, double* lx,
	int type1, int type2, int atom_i, int atom_j,
	double* fi, double* fj, double* fk, double* fl,
	int eflag)
{
	// Map neighbor atoms to closest image
	double jximage[3], lximage[3];
	domain->closest_image(ix, jx, jximage);
	domain->closest_image(kx, lx, lximage);
	jx = &(jximage[0]);
	lx = &(lximage[0]);

	// Diameter a
	const double a = 2*radius[type1][type2];
	const double R = radius[type1][type2];


	const unsigned int num_sample_points = 3;
	const unsigned int num_closest_points = 1;
	const unsigned int max_points = num_sample_points + num_closest_points;


	double x_points[max_points][3], y_points[max_points][3];
	unsigned int num_xpoints = num_sample_points;
	unsigned int num_ypoints = num_sample_points;

	for (unsigned int i = 0; i < 3; i++)
	{
		// Copy in x sample points
		x_points[0][i] = ix[i];
		x_points[1][i] = jx[i];
		x_points[2][i] = (ix[i] + jx[i]) / 2;

		// Copy in y sample points
		y_points[0][i] = kx[i];
		y_points[1][i] = lx[i];
		y_points[2][i] = (kx[i] + lx[i]) / 2;
	}

	// Determine closest point
	double rv[3], nx[3], ny[3];
	for (unsigned int i = 0; i < 3; i++)
	{
		rv[i] = y_points[0][i] - x_points[0][i];
		nx[i] = x_points[1][i] - x_points[0][i];
		ny[i] = y_points[1][i] - y_points[0][i];
	}

	double L1 = norm(nx);
	double L2 = norm(ny);

	scale(1/L1, nx);
	scale(1/L2, ny);

	double cos_t = dot(nx, ny);
	double cos2_t_1 = cos_t * cos_t - 1;

	double x0, y0;

	if (fabs(cos2_t_1) < 0.05)
	{
		// Parallel rods
		x0 = dot(rv, nx);
		y0 = 0;
	}
	else
	{
		// Skew rods
		double rx = dot(rv,nx);
		double ry = dot(rv,ny);
		x0 = (ry*cos_t - rx)/cos2_t_1;
		y0 = (rx*cos_t - ry)/cos2_t_1;
	}

	// Add closest point to the arrays
	if ((R < x0 and x0 < L1/2-R) or (L1/2+R < x0 and x0 < L1-R))
	{
		for (unsigned int i = 0; i < 3; i++)
		{
			x_points[3][i] = x_points[0][i] + x0 * nx[i];
		}
		num_xpoints++;
	}

	if ((R < y0 and y0 < L2/2-R) or (L2/2+R < y0 and y0 < L2-R))
	{
		for (unsigned int i = 0; i < 3; i++)
		{
			y_points[3][i] = y_points[0][i] + y0 * ny[i];
		}
		num_ypoints++;
	}

	// Calculate colloidal interactions
	double delx, dely, delz, rsq, r;
	double fR, dUR, dUA, evdwl, fpair;
	double K[9], g[4], h[4];

	double forces[max_points][3], torques[max_points][3];
	for (unsigned int i = 0; i < num_xpoints; i++)
	{
		forces[i][0] = 0;
		forces[i][1] = 0;
		forces[i][2] = 0;
		torques[i][0] = 0;
		torques[i][1] = 0;
		torques[i][2] = 0;
	}

	double factor_lj = force->special_lj[sbmask(atom_j)];

	// Iterate over arrays and compute interactions
	for (unsigned int ii = 0; ii < num_xpoints; ii++)
	{
		for (unsigned int jj = 0; jj < num_ypoints; jj++)
		{
			// Colloidal interaction
			delx = y_points[jj][0] - x_points[ii][0];
			dely = y_points[jj][1] - x_points[ii][1];
			delz = y_points[jj][2] - x_points[ii][2];
			rsq = delx*delx + dely*dely + delz*delz;
			r = sqrt(rsq);

			K[0] = R*R;
			K[1] = 2*R;
			K[2] = 0;
			K[3] = K[1]+r;
			K[4] = K[1]-r;
			K[5] = K[2]+r;
			K[6] = K[2]-r;
			K[7] = 1.0/(K[3]*K[4]);
			K[8] = 1.0/(K[5]*K[6]);
			g[0] = powint(K[3],-7);
			g[1] = powint(K[4],-7);
			g[2] = powint(K[5],-7);
			g[3] = powint(K[6],-7);
			h[0] = ((K[3]+5.0*K[1])*K[3]+30.0*K[0])*g[0];
			h[1] = ((K[4]+5.0*K[1])*K[4]+30.0*K[0])*g[1];
			h[2] = ((K[5]+5.0*K[2])*K[5]-30.0*K[0])*g[2];
			h[3] = ((K[6]+5.0*K[2])*K[6]-30.0*K[0])*g[3];
			g[0] *= 42.0*K[0]/K[3]+6.0*K[1]+K[3];
			g[1] *= 42.0*K[0]/K[4]+6.0*K[1]+K[4];
			g[2] *= -42.0*K[0]/K[5]+6.0*K[2]+K[5];
			g[3] *= -42.0*K[0]/K[6]+6.0*K[2]+K[6];

			fR = vdw[type1][type2]*sigma6[type1][type2]/r/37800.0;
			evdwl = fR * (h[0]-h[1]-h[2]+h[3]);
			dUR = evdwl/r + 5.0*fR*(g[0]+g[1]-g[2]-g[3]);
			dUA = -vdw[type1][type2]/3.0*r*((2.0*K[0]*K[7]+1.0)*K[7] +
							(2.0*K[0]*K[8]-1.0)*K[8]);
			fpair = factor_lj * (dUR+dUA)/r;
			if (eflag)
			  evdwl += vdw[type1][type2]/6.0 *
			    (2.0*K[0]*(K[7]+K[8])-log(K[8]/K[7]));
			if (r <= K[1])
			  error->one(FLERR,"Overlapping large/large in pair colloid");

			if (eflag) evdwl *= factor_lj;

			forces[ii][0] += delx*fpair;
			forces[ii][1] += dely*fpair;
			forces[ii][2] += delz*fpair;

			torques[ii][0] += (x_points[ii][1]-x_points[2][1])*forces[ii][2] - (x_points[ii][2]-x_points[2][2])*forces[ii][1];
			torques[ii][1] += (x_points[ii][2]-x_points[2][2])*forces[ii][0] - (x_points[ii][0]-x_points[2][0])*forces[ii][2];
			torques[ii][2] += (x_points[ii][0]-x_points[2][0])*forces[ii][1] - (x_points[ii][1]-x_points[2][1])*forces[ii][0];


			if (evflag) ev_tally(atom_i, atom_j, atom->nlocal, force->newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
		}
	}

	// Calculate net force and torque on rod
	double F[3], tau[3];
	for (unsigned int i = 0; i < 3; i++)
	{
		F[i] = 0;
		tau[i] = 0;
		for (unsigned int j = 0; j < num_xpoints; j++)
		{
			F[i] += forces[j][i];
			tau[i] += torques[j][i];
		}
	}

	// Project net force and torque to forces on endpoints
	double f[3];
	cross(tau, nx, f);
	double tnx = norm(f);
	if (tnx < 0.05)
	{
		scale(0, f);
	}
	else
	{
		scale(2*norm(tau)/(L1*tnx), f);
	}

	if (norm(F) > 5)
	{
		printf("LARGE FORCES\n");
		printf("ix: <%f, %f, %f>\n", ix[0], ix[1], ix[2]);
		printf("jx: <%f, %f, %f>\n", jx[0], jx[1], jx[2]);
		printf("kx: <%f, %f, %f>\n", kx[0], kx[1], kx[2]);
		printf("lx: <%f, %f, %f>\n", lx[0], lx[1], lx[2]);
		printf("F: <%f, %f, %f>\n", F[0], F[1], F[2]);
		printf("f: <%f, %f, %f>\n", f[0], f[1], f[2]);
		printf("x_points:\n");
		for (unsigned int i = 0; i < num_xpoints; i++)
		{
			printf("%d: <%f, %f, %f>\n", i, x_points[i][0], x_points[i][1], x_points[i][2]);
		}
		printf("y_points:\n");
		for (unsigned int i = 0; i < num_ypoints; i++)
		{
			printf("%d: <%f, %f, %f>\n", i, y_points[i][0], y_points[i][1], y_points[i][2]);
		}

	}

	for (unsigned int i = 0; i < 3; i++)
	{
		fi[i] += (F[i] + f[i]) / 2;
		fj[i] += (F[i] - f[i]) / 2;
		fk[i] -= (F[i] + f[i]) / 2;
		fl[i] -= (F[i] - f[i]) / 2;
	}

	//printf("Tube 1 endpoint 1: <%f, %f, %f>\n", ix[0], ix[1], ix[2]);
	//printf("Tube 2 endpoint 2: <%f, %f, %f>\n", jx[0], jx[1], jx[2]);
	//printf("Net force on tube 1: <%f, %f, %f>\n", F[0], F[1], F[2]);
}
