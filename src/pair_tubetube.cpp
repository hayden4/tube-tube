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

/**
 * Sets flags for tube-tube potential
 */
PairTubeTube::PairTubeTube(LAMMPS *lmp) : Pair(lmp)
{
	respa_enable = 0;
	writedata = 1;
}

/* ---------------------------------------------------------------------- */

/**
 * Destroys any memory that has been allocated
 */
PairTubeTube::~PairTubeTube()
{
	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(cut);
		memory->destroy(hamaker);
		memory->destroy(radius);
		memory->destroy(vdw);
		memory->destroy(xi);
	}
}

/* ---------------------------------------------------------------------- */

/**
 * This method is called every integration step to calculate
 * forces and energy for local atoms.
 */
void PairTubeTube::compute(int eflag, int vflag)
{
	// Define local variables to shorten names
	double** x = atom->x;

	// Initialize EV tallying
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

				// Check if k will be iterated over in this compute step
				if (nl_contains(atom->tag[k], list->ilist, list->inum))
				{
					// If so, get its local index
					k = atom->map(atom->tag[k]);

					// Iterate over k's bonded atoms
					for (int ll = 0; ll < atom->num_bond[k]; ll++)
					{
						// Get lth atom bonded to k
						int l = atom->map(atom->bond_atom[k][ll]);

						// i is local
						// k is local

						// Perform double counting check
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

						// Calculate the forces on the current permutation of atoms
						calculate_force(atom->x[i], atom->x[j], atom->x[k], atom->x[l], type1, type2, i, k, atom->f[i], atom->f[j], atom->f[k], atom->f[l]);
					}
				}
				else
				{
					// If k is local, skip it
					if (k < atom->nlocal)
						continue;

					// Get k's local index
					k = atom->map(atom->tag[k]);

					// Iterate over k's ghost bonded atoms
					for (int ll = 0; ll < atom->nbond_max; ll++)
					{
						// Get l's tag
						tagint tag_l = atom->ghost_bond_tags[k-atom->nlocal][ll];

						// Check if l is a valid tag
						if (tag_l <= 0) continue;

						int l = atom->map(tag_l);

						// i is local
						// k is not local
						
						// Perform double counting check
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

						// Calculate forces on the current permutation of atoms
						calculate_force(atom->x[i], atom->x[j], atom->x[k], &atom->ghost_bond_x[k-atom->nlocal][3*ll], type1, type2, i, k, atom->f[i], atom->f[j], atom->f[k], &atom->ghost_bond_f[k-atom->nlocal][3*ll]);
					}
				}
			}
		}
	}

	// Perform virial computations post compute
	if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

/**
 * Allocate parameter arrays
 */
void PairTubeTube::allocate()
{
	// Set allocated to true
	allocated = 1;
	int n = atom->ntypes;
	
	// Allocate set flag array
	memory->create(setflag,n+1,n+1,"pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;
	
	// Allocate parameter arrays
	memory->create(cutsq,n+1,n+1,"pair:cutsq");
	memory->create(cut,n+1,n+1,"pair:cut");
	memory->create(hamaker,n+1,n+1,"pair:hamaker");
	memory->create(radius,n+1,n+1,"pair:radius");
	memory->create(xi,n+1,n+1,"pair:xi");
	memory->create(vdw,n+1,n+1,"pair:vdw");
}

/* ---------------------------------------------------------------------- */

/**
 * Read settings command from input script
 */
void PairTubeTube::settings(int narg, char **arg)
{
	// Expect 2 arguments exactly: repulsion type and cutoff
	if (narg != 2) error->all(FLERR,"Illegal pair_style command");
// Repulsion types:
	// none - NO_REPULSION
	// power - POWER_REPULSION
	// exp - EXP_REPULSION
	if (strcmp(arg[0],"none") == 0) {
		repulsion_type = NO_REPULSION;
	} else if (strcmp(arg[0],"power") == 0) {
		repulsion_type = POWER_REPULSION;
	} else if (strcmp(arg[0],"exp") == 0) {
		repulsion_type = EXP_REPULSION;
	} else {
		error->all(FLERR, "Illegal pair tubetube command");
	}

	// Read global cutoff
	cut_global = force->numeric(FLERR,arg[1]);

	// If parameter arrays are already allocated, set the new cutoff
	if (allocated) {
		for (int i = 1; i <= atom->ntypes; i++)
			for (int j = i; j <= atom->ntypes; j++)
				if (setflag[i][j]) cut[i][j] = cut_global;
	}
}

/* ---------------------------------------------------------------------- */

/**
 * Load coefficients from an input script
 */
void PairTubeTube::coeff(int narg, char **arg)
{
	// Expecting 4 to 6 arguments
	if (narg < 4 || narg > 6)
		error->all(FLERR, "Incorrect args for pair coefficients");

	// Make sure parameter arrays are allocated
	if (!allocated) allocate();

	// Get atom types to set
	int ilo,ihi,jlo,jhi;
	force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
	force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

	// Read hamaker constant and tube radius
	double hamaker_one = force->numeric(FLERR, arg[2]);
	double radius_one  = force->numeric(FLERR, arg[3]);

	// Read repulsion constant if available
	double xi_one = 0;
	if (narg >= 5) xi_one = force->numeric(FLERR, arg[4]);
	
	// Read type specific cutoff if available
	double cut_one = cut_global;
	if (narg >= 6) cut_one = force->numeric(FLERR, arg[5]);

	// Set parameters for given types
	int count = 0;
	for (int i = ilo; i <= ihi; i++) {
		for (int j = MAX(jlo,i); j <= jhi; j++) {
			hamaker[i][j] = hamaker_one;
			radius[i][j] = radius_one;
			xi[i][j] = xi_one;
			cut[i][j] = cut_one;
			setflag[i][j] = 1;
			count++;
		}
	}

	if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

/**
 * Writes pair data to the file
 */
void PairTubeTube::write_data(FILE *fp)
{
	for (int i = 1; i <= atom->ntypes; i++)
		fprintf(fp, "%d %g %g %g\n", i, hamaker[i][i], radius[i][i], xi[i][i]);
}

/* ---------------------------------------------------------------------- */

/**
 * Writes pair data to the file
 */
void PairTubeTube::write_data_all(FILE *fp)
{
	for (int i = 1; i <= atom->ntypes; i++)
		for (int j = 1; j <= atom->ntypes; j++)
			fprintf(fp, "%d %d %g %g %g %g\n", i, j, hamaker[i][j], radius[i][j], xi[i][j], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

/**
 * Writes coefficients to a restart file
 */
void PairTubeTube::write_restart(FILE *fp)
{
	write_restart_settings(fp);

	for (int i = 1; i <= atom->ntypes; i++)
	{
		for (int j = i; j <= atom->ntypes; j++)
		{
			fwrite(&setflag[i][j], sizeof(int), 1, fp);
			if (setflag[i][j]) {
				fwrite(&hamaker[i][j], sizeof(double), 1, fp);
				fwrite(&radius[i][j], sizeof(double), 1, fp);
				fwrite(&xi[i][j], sizeof(double), 1, fp);
				fwrite(&cut[i][j], sizeof(double), 1, fp);
			}
		}
	}
}

/* ---------------------------------------------------------------------- */

/**
 * Read coefficients from a restart file
 */
void PairTubeTube::read_restart(FILE *fp)
{
	read_restart_settings(fp);
	allocate();

	for (int i = 1; i <= atom->ntypes; i++)
	{
		for (int j = 1; j <= atom->ntypes; j++)
		{
			if (comm->me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
			MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
			if (setflag[i][j]) {
				if (comm->me == 0) {
					utils::sfread(FLERR,&hamaker[i][j], sizeof(double), 1, fp, nullptr, error);
					utils::sfread(FLERR,&radius[i][j], sizeof(double), 1, fp, nullptr, error);
					utils::sfread(FLERR,&xi[i][j], sizeof(double), 1, fp, nullptr, error);
					utils::sfread(FLERR,&cut[i][j], sizeof(double), 1, fp, nullptr, error);
				}
				MPI_Bcast(&hamaker[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&radius[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&xi[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
			}
		}
	}
}

/* ---------------------------------------------------------------------- */

/**
 * Write settings to a restart file
 */
void PairTubeTube::write_restart_settings(FILE *fp)
{
	fwrite(&cut_global, sizeof(double), 1, fp);
	fwrite(&repulsion_type, sizeof(int), 1, fp);
}

/* ---------------------------------------------------------------------- */

/**
 * Read settings from a restart file
 */
void PairTubeTube::read_restart_settings(FILE *fp)
{
	if (comm->me == 0)
	{
		utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
		utils::sfread(FLERR, &repulsion_type, sizeof(int), 1, fp, nullptr, error);
	}

	MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
	MPI_Bcast(&repulsion_type, 1, MPI_INT, 0, world);
}

/* ---------------------------------------------------------------------- */

/**
 * Initializes the interaction between atom types i and j
 */
double PairTubeTube::init_one(int i, int j)
{
	// Calculate vdw as the constant prefactor of the potential energy equation
	vdw[i][j] = hamaker[i][j] * M_PI * pow(2.0*radius[i][j], 4.0) / 32.0;
	vdw[j][i] = vdw[i][j];

	return cut[i][j];
}



/* ---------------------------------------------------------------------- */

/**
 * Iterates over the neighborlist ls of length size, and
 * checks to see if it contains the tag a. The neighborlist should
 * contain local ids (ints).
 * Returns true if ls contains a, false otherwise.
 */
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
/* ---------------------------------------------------------------------- */

// Vector operations

/**
 * Computes the dot product of two 3D vectors a
 * and b.
 */
double dot(double* a, double* b)
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * Computes the norm of a 3D vector a
 */
double norm(double* a)
{
	return sqrt(dot(a,a));
}

/**
 * Computes the cross product of 3D vectors a and b
 * and places the result in c
 */
void cross(double* a, double* b, double* c)
{
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
}

/**
 * Multiplies all elements a 3D vector v by a
 */
void scale(double a, double* v)
{
	for (int i = 0; i < 3; i++)
		v[i] *= a;
}

/**
 * Computes the shortest distance from a point p to a line defined by
 * f(t) = p1 + n*t
 * where t in [0, L]
 */
double point_to_segment(double* p, double* p1, double* n, double L)
{
	double lambda = (n[0]*(p[0]-p1[0]) + n[1]*(p[1]-p1[1]) + n[2]*(p[2]-p1[2]));
	if (lambda < 0) lambda = 0;
	if (lambda > L) lambda = L;

	double q[3];
	for (int i = 0; i < 3; i++)
		q[i] = p1[i] + lambda * n[i] - p[i];
	
	return norm(q);
}

/* ---------------------------------------------------------------------- */

// Math functions for Logan-Tkachenko potential

/**
 * Computes the minimum of two numbers
 */
double min(double a, double b)
{
	return (a < b) ? a : b;
}

/**
 * g function defined in Logan-Tkachenko
 */
double g(double x)
{
	double s = (x < 0) ? -1 : 1;
	return 0.5 * s * min(1, 1.5*fabs(x));
}

/**
 * gamma function defined in Logan-Tkachenko
 */
double gamma(double xp, double xm, double yp, double ym)
{
	return min(g(xp) - g(xm), g(yp) - g(ym));
}

/**
 * Heaviside function
 * Returns 0 if x < 0
 * Else returns 1
 */
double heaviside(double x)
{
	return (x < 0) ? 0 : 1;
}

/**
 * Compute the sign of a number
 */
double sgn(double x)
{
	return (x < 0) ? -1 : 1;
}

/* ---------------------------------------------------------------------- */

/**
 * Calculates forces between tube 1 (ix, jx) and tube 2
 * (kx, lx). The type of the tubes are type1 and type2
 * respectively. atom_i and atom_j refer to the indexes
 * of the lead atom in tube (used for energy tallying).
 * Forces are added to the vectors (fi, fj) for tube 1
 * and (fk, fl) for tube 2.
 */
void PairTubeTube::calculate_force(
	double* ix, double* jx, double* kx, double* lx,
	int type1, int type2, int atom_i, int atom_j,
	double* fi, double* fj, double* fk, double* fl
	)
{
	double jximage[3], lximage[3];
	domain->closest_image(ix, jx, jximage);
	domain->closest_image(kx, lx, lximage);
	jx = &(jximage[0]);
	lx = &(lximage[0]);

	// Diameter a
	const double a = 2*radius[type1][type2];

	double nxp[3], nyp[3];
	double c[3], *nx, *ny, nz[3];

	nx = nxp;
	ny = nyp;

	// Calculate c = Xc - Yc
	// nx unnormalized
	// ny unnormalized
	for (int i = 0; i < 3; i++)
	{
		c[i] = (ix[i] + jx[i] - kx[i] - lx[i]) / 2;
		nx[i] = jx[i] - ix[i];
		ny[i] = lx[i] - kx[i];
	}

	// Lengths of each tube
	const double L1 = norm(nx);
	const double L2 = norm(ny);

	// Single length (arithmetic or geometric)
	const double L = (L1 + L2) / 2;

	// Normalize nx and ny
	scale(1/L1, nx);
	scale(1/L2, ny);

	// Calculate nz
	cross(nx, ny, nz);
	scale(sgn(dot(c,nz)), nz);
	
	// Calculate cos_t and sin_t
	const double cos_t = dot(nx, ny);
	const double sin_t = norm(nz);
	const double acos_t = fabs(cos_t);
	const double asin_t = fabs(sin_t);

	// Define x0 and y0
	double x0 = dot(nx, c)/2;
	double y0 = dot(ny, c)/2;

	// Adjust nz if parallel rods
	// Otherwise calculate x0 and y0
	if (asin_t < 0.05)
	{
		const double c_norm = norm(c);
		nz[0] = c[0] / c_norm;
		nz[1] = c[1] / c_norm;
		nz[2] = c[2] / c_norm;
	}
	else
	{
		const double cnx = dot(c, nx);
		const double cny = dot(c, ny);
		x0 =  (cny*cos_t - cnx) / (sin_t*sin_t);
		y0 = -(cnx*cos_t - cny) / (sin_t*sin_t);
		scale(1/asin_t, nz);
	}

	// Calculate r
	double r = fabs(dot(c, nz));
	if (r < a)
	{
		const double abc = point_to_segment(kx, ix, nx, L1);
		const double abd = point_to_segment(lx, ix, nx, L1);
		const double cda = point_to_segment(ix, kx, ny, L2);
		const double cdb = point_to_segment(jx, kx, ny, L2);

		r = min(min(abc,abd), min(cda,cdb));
	}

	// If |x0| > |y0|, swap tubes
	if (fabs(x0) > fabs(y0))
	{
		double t = x0;
		x0 = y0;
		y0 = t;

		double* temp = ix;
		ix = kx;
		kx = temp;

		temp = jx;
		jx = lx;
		lx = temp;

		temp = fi;
		fi = fk;
		fk = temp;

		temp = fj;
		fj = fl;
		fl = temp;

		temp = nx;
		nx = ny;
		ny = temp;

		scale(-1, nz);

		int swap = atom_i;
		atom_i = atom_j;
		atom_j = swap;
	}


	/* ---------------------------------------- */
	/* Begin force calculation section          */
	/* ---------------------------------------- */
	double F1[3], F2[3], tau1[3], tau2[3];

	const double v = acos_t + 3*L*asin_t / (4*(r+a));

	const double xp = 4*(v*(x0 + L/2) - y0*cos_t) / (3*L);
	const double xm = 4*(v*(x0 - L/2) - y0*cos_t) / (3*L);
	const double yp = 4*(v*(y0 + L/2) - x0*cos_t) / (3*L);
	const double ym = 4*(v*(y0 - L/2) - x0*cos_t) / (3*L);

	const double xap = (x0 + L/2)/a + (L-y0*cos_t)/(2*a*v) - 1.4;
	const double xam = (x0 - L/2)/a - (L+y0*cos_t)/(2*a*v) + 1.4;
	const double yap = (y0 + L/2)/a + (L-x0*cos_t)/(2*a*v) - 1.4;
	const double yam = (y0 - L/2)/a - (L+x0*cos_t)/(2*a*v) + 1.4;

	const double r_eff = r - gamma(xap, xam, yap, yam) * a;
	if (r_eff <= 0)
	{
		printf("ERROR: r_eff < 0  -->  tubes overlap!\n");
		printf("\tL1: %f\n", L1);
		printf("\tL2: %f\n", L2);
		return;
	}
	const double eta = 1 - (xi[type1][type2] * xi[type1][type2]) / (r_eff * r_eff);
	const double uvdw = - vdw[type1][type2] * gamma(xp,xm,yp,ym) / (r_eff * (asin_t + 2.35*sqrt(r*r_eff)/L)*(r+0.12*a)*(r+0.12*a)*(r+0.12*a));
	const double utotal = uvdw * eta;


	const double lambda = (1.0 + 1.0/(2.0 + L*asin_t/a))/r_eff;
	const double sigma = 2*xi[type1][type2]*xi[type1][type2] / (r_eff*r_eff*r_eff);
	const double dgadr = 9*L*asin_t*(heaviside(2.0/3.0 - fabs(yap))*(L-2*x0*cos_t) - heaviside(2.0/3.0 - fabs(yam))*(L+2*x0*cos_t)) / (32*a*v*v*(r+a)*(r+a));
	const double P = sigma*(1.0 - a*dgadr) - eta*(lambda + 3.0/(r+0.12*a));

	double phi_0 = 0;

	double xc = -x0;
	double yc = -y0;

	if (fabs(xc) > L/2)
	{
		xc = copysign(L/2, xc);
		yc = xc * cos_t - dot(c, nx);
	}

	if (fabs(yc) > L/2)
	{
		yc = copysign(L/2, yc);
		xc = yc * cos_t + dot(c, ny);
	}

	double force_dir[3];
	force_dir[0] = c[0] + yc * ny[0] - xc * nx[0];
	force_dir[1] = c[1] + yc * ny[1] - xc * nx[1];
	force_dir[2] = c[2] + yc * ny[2] - xc * nx[2];


	if (asin_t < 0.05)
	{
		for (int i = 0; i < 3; i++)
		{
			F1[i] = -uvdw * P * force_dir[i];
			F2[i] = -F1[i];
		}

		double snx[3], sny[3];
		for (int i = 0; i < 3; i++)
		{
			snx[i] = -x0 * nx[i];
			sny[i] =  y0 * ny[i];
		}
		cross(snx, F1, tau1);
		cross(sny, F2, tau2);
	}
	else
	{
		const double phi1 = 3*(eta*lambda - sigma) * (heaviside(2.0/3.0 - fabs(yap)) - heaviside(2.0/3.0 - fabs(yam))) / (4*v);
		const double phi2 = eta * (heaviside(2.0/3.0 - fabs(yp)) - heaviside(2.0/3.0 - fabs(ym))) / (lambda*L);
		const double phi = (phi1 + phi2) / (sin_t*sin_t);
		phi_0 = phi;
		for (int i = 0; i < 3; i++)
		{
			//const double f = (1+v)*cos_t*nx[i] + (v + cos_t*cos_t)*ny[i];
			//F1[i] = uvdw*(phi*f - P*nz[i]);
			F1[i] = -uvdw*P*force_dir[i];
			F2[i] = -F1[i];
		}

		const double dvdt = -sgn(cos_t)*sin_t + sgn(sin_t)*3*L*cos_t / (4*(r+a));
		const double dgadt = 3*(heaviside(2.0/3.0-fabs(yap))*(y0*(v*sin_t + dvdt*cos_t) - dvdt*L) - heaviside(2.0/3.0-fabs(yam))*(y0*(v*sin_t + dvdt*cos_t) + dvdt*L)) / (8*a*v*v);
		const double factor = uvdw*(sigma*dgadt + eta*cos_t/(sin_t*(1+2.35*sqrt(r*r_eff)/(L*asin_t))));
		double snx[3], sny[3];
		for (int i = 0; i < 3; i++)
		{
			snx[i] = -x0 * nx[i];
			sny[i] =  y0 * ny[i];
		}
		cross(snx, F1, tau1);
		cross(sny, F2, tau2);
		for(int i = 0; i < 3; i++)
		{
			//tau1[i] -= factor * nz[i];
			//tau2[i] += factor * nz[i];
		}
	}
	

	/* ---------------------------------------- */
	/* End Force Calculation Section            */
	/* ---------------------------------------- */


	// Apply force and torque to atoms
	double f1[3], f2[3];
	cross(tau1, nx, f1);
	cross(tau2, ny, f2);

	const double t1 = norm(tau1);
	const double t2 = norm(tau2);

	const double tn1 = norm(f1);
	const double tn2 = norm(f2);

	if (tn1 <= 0.005*t1)
	{
		scale(0, f1);
	}
	else
	{
		scale(2*t1/(L1*tn1), f1);
	}

	if (tn2 <= 0.005*t2)
	{
		scale(0, f2);
	}
	else
	{
		scale(2*t2/(L2*tn2), f2);
	}

	// ----- DEBUG ---- //
	
	const double large = 100;

	if (norm(F1) > large or norm(f1) > large) {
		printf("LARGE FORCES:\n");
		printf("i %d, k %d\n", atom_i, atom_j);
		printf("ix: <%f, %f, %f>\n", ix[0], ix[1], ix[2]);
		printf("jx: <%f, %f, %f>\n", jx[0], jx[1], jx[2]);
		printf("kx: <%f, %f, %f>\n", kx[0], kx[1], kx[2]);
		printf("lx: <%f, %f, %f>\n", lx[0], lx[1], lx[2]);
		printf("L1: %f\n", L1);
		printf("L2: %f\n", L2);
		printf("(x0, y0, sint, cost, r): (%f, %f, %f, %f, %f)\n", x0, y0, sin_t, cos_t, r);
		printf("nx: <%f, %f, %f>\n", nx[0], nx[1], nx[2]);
		printf("ny: <%f, %f, %f>\n", ny[0], ny[1], ny[2]);
		printf("nz: <%f, %f, %f>\n", nz[0], nz[1], nz[2]);

		printf("F1: <%f, %f, %f>\n", F1[0], F1[1], F1[2]);
		printf("f1: <%f, %f, %f>\n", f1[0], f1[1], f1[2]);
		printf("uvdw: %f\n", uvdw);
		printf("phi: %f\n", phi_0);
		printf("P: %f\n", P);
		printf("lambda: %f\n", lambda);
		printf("sigma: %f\n", sigma);
		printf("r_eff: %f\n", r_eff);
		printf("vdw: %f\n", vdw[type1][type2]);

		printf("\n");
	}

	// ----- END DEBUG ----- //

	for (int i = 0; i < 3; i++)
	{
		fi[i] += (F1[i] - f1[i]) / 2;
		fj[i] += (F1[i] + f1[i]) / 2;
		fl[i] += (F2[i] - f2[i]) / 2;
		fk[i] += (F2[i] + f2[i]) / 2;
	}

	// Tally energy calculations
	if (evflag)
		ev_tally(atom_i, atom_j, atom->nlocal, force->newton_pair, utotal, 0.0, sgn(dot(c, F1)) * norm(F1), c[0], c[1], c[2]);
}
