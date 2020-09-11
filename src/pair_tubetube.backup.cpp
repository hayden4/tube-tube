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
		memory->destroy(hamaker);
		memory->destroy(radius);
		memory->destroy(vdw);
		memory->destroy(xi);
	}
}

/* ---------------------------------------------------------------------- */

void PairTubeTube::compute(int eflag, int vflag)
{
	double** x = atom->x;

	checked_atoms.clear();
	this_ones.clear();
	potentials.clear();

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
			int j = domain->closest_image(i, atom->map(atom->bond_atom[i][jj]));

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
						int l = domain->closest_image(k, atom->map(atom->bond_atom[k][ll]));

						// i is local
						// k is local

						int choice = 1;
						if (nl_contains(atom->tag[l], list->firstneigh[i], list->numneigh[i])) if (atom->tag[l] < atom->tag[k]) choice = 0;
						if (nl_contains(atom->tag[i], list->firstneigh[k], list->numneigh[k])) if (atom->tag[k] < atom->tag[i]) choice = 0;
						if (nl_contains(atom->tag[j], list->firstneigh[k], list->numneigh[k])) if (atom->tag[k] < atom->tag[i]) choice = 0;
						if (nl_contains(atom->tag[j], list->ilist, list->inum))
						{
							if (nl_contains(atom->tag[k], list->firstneigh[j], list->numneigh[j])) if (atom->tag[j] < atom->tag[i]) choice = 0;
							if (nl_contains(atom->tag[l], list->firstneigh[j], list->numneigh[j])) if (atom->tag[j] < atom->tag[i]) choice = 0;
						}
						if (nl_contains(atom->tag[l], list->ilist, list->inum))
						{
							if (nl_contains(atom->tag[i], list->firstneigh[l], list->numneigh[l])) if (atom->tag[l] < atom->tag[i]) choice = 0;
							if (nl_contains(atom->tag[j], list->firstneigh[l], list->numneigh[l])) if (atom->tag[l] < atom->tag[i]) choice = 0;
						}

						checked_atoms.push_back(std::vector<tagint>());
						checked_atoms.back().push_back(atom->tag[i]);
						checked_atoms.back().push_back(atom->tag[j]);
						checked_atoms.back().push_back(atom->tag[k]);
						checked_atoms.back().push_back(atom->tag[l]);

						this_ones.push_back(choice);

						if (choice == 1)
							potentials.push_back(calculate_force(atom->x[i], atom->x[j], atom->x[k], atom->x[l], type1, type2, atom->f[i], atom->f[j], atom->f[k], atom->f[l]));
						else
							potentials.push_back(0);
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

						int choice = 1;
						if (nl_contains(atom->tag[l], list->firstneigh[i], list->numneigh[i])) if (atom->tag[l] < atom->tag[k]) choice = 0;
						if (nl_contains(atom->tag[j], list->ilist, list->inum))
						{
							if (nl_contains(atom->tag[k], list->firstneigh[j], list->numneigh[j])) if (atom->tag[j] < atom->tag[i]) choice = 0;
							if (nl_contains(atom->tag[l], list->firstneigh[j], list->numneigh[j])) if (atom->tag[j] < atom->tag[i]) choice = 0;
						}
						if (l > -1 and nl_contains(atom->tag[l], list->ilist, list->inum))
						{
							if (nl_contains(atom->tag[i], list->firstneigh[l], list->numneigh[l])) if (atom->tag[l] < atom->tag[i]) choice = 0;
							if (nl_contains(atom->tag[j], list->firstneigh[l], list->numneigh[l])) if (atom->tag[l] < atom->tag[i]) choice = 0;
						}



						checked_atoms.push_back(std::vector<tagint>());
						checked_atoms.back().push_back(atom->tag[i]);
						checked_atoms.back().push_back(atom->tag[j]);
						checked_atoms.back().push_back(atom->tag[k]);
						checked_atoms.back().push_back(      tag_l );

						this_ones.push_back(choice);

						if (choice == 1)
							potentials.push_back(calculate_force(atom->x[i], atom->x[j], atom->x[k], &atom->ghost_bond_x[k-atom->nlocal][3*ll], type1, type2, atom->f[i], atom->f[j], atom->f[k], &atom->ghost_bond_f[k-atom->nlocal][3*ll]));
						else
							potentials.push_back(0);
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
	memory->create(hamaker,n+1,n+1,"pair:hamaker");
	memory->create(radius,n+1,n+1,"pair:radius");
	memory->create(xi,n+1,n+1,"pair:xi");
	memory->create(vdw,n+1,n+1,"pair:vdw");
}

/* ---------------------------------------------------------------------- */

void PairTubeTube::settings(int narg, char **arg)
{
	if (narg != 2) error->all(FLERR,"Illegal pair_style command");

	if (strcmp(arg[0],"none") == 0) {
		repulsion_type = NO_REPULSION;
	} else if (strcmp(arg[0],"power") == 0) {
		repulsion_type = POWER_REPULSION;
	} else if (strcmp(arg[0],"exp") == 0) {
		repulsion_type = EXP_REPULSION;
	} else {
		error->all(FLERR, "Illegal pair tubetube command");
	}

	cut_global = force->numeric(FLERR,arg[1]);

	if (allocated) {
		for (int i = 1; i <= atom->ntypes; i++)
			for (int j = i; j <= atom->ntypes; j++)
				if (setflag[i][j]) cut[i][j] = cut_global;
	}
}

/* ---------------------------------------------------------------------- */

void PairTubeTube::coeff(int narg, char **arg)
{
	//printf("COEFF CALLED\n");

	if (narg < 4 || narg > 6)
		error->all(FLERR, "Incorrect args for pair coefficients");

	if (!allocated) allocate();

	int ilo,ihi,jlo,jhi;
	force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
	force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

	double hamaker_one = force->numeric(FLERR, arg[2]);
	double radius_one  = force->numeric(FLERR, arg[3]);

	double xi_one = 0;
	if (narg >= 5) xi_one = force->numeric(FLERR, arg[4]);
	
	double cut_one = cut_global;
	if (narg >= 6) cut_one = force->numeric(FLERR, arg[5]);

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

void PairTubeTube::write_data(FILE *fp)
{
	fprintf(fp, "Proc %d\n", comm->me);
	fprintf(fp, "THIS IS THE TUBE TUBE POTENTIAL\n");

	fprintf(fp, "Neighbor Lists:\n");
	for (int ii = 0; ii < list->inum; ii++)
	{
		int i = list->ilist[ii];
		fprintf(fp, "%4d:\t", atom->tag[i]);

		for (int jj = 0; jj < list->numneigh[i]; jj++)
		{
			int j = list->firstneigh[i][jj];
			j &= NEIGHMASK;

			fprintf(fp, "%4d ", atom->tag[j]);
		}

		fprintf(fp, "\n");
	}

	fprintf(fp, "vdw\n\t");
	for (int j = 1; j <= atom->ntypes; j++)
	{
		fprintf(fp, "%1d\t", j);
	}
	fprintf(fp, "\n");

	for (int i = 1; i <= atom->ntypes; i++)
	{
		fprintf(fp, "%1d\t", i);
		for (int j = 1; j <= atom->ntypes; j++)
		{
			fprintf(fp, "%f\t", vdw[i][j]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\nchecked_atoms:\n");
	for (int i = 0; i < checked_atoms.size(); i++)
	{
		fprintf(fp, "%4d %4d\t%4d %4d", checked_atoms[i][0], checked_atoms[i][1], checked_atoms[i][2], checked_atoms[i][3]);
		fprintf(fp, "\t\t%1d\t%f\n", this_ones[i], potentials[i]);
	}
}

void PairTubeTube::write_data_all(FILE *fp)
{
	fprintf(fp, "THIS IS THE TUBE TUBE POTENTIAL ALL\n");
}

double PairTubeTube::init_one(int i, int j)
{
	//printf("INIT ONE CALLED\n");

	/*
	if (setflag[i][j] == 0) {
		// This needs to be checked :(
		hamaker[i][j] = mix_energy(hamaker[i][i], hamaker[j][j], radius[i][i], radius[j][j]);
		radius[i][j] = mix_distance(radius[i][i], radius[j][j]);
		cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
	}
	*/

	vdw[i][j] = hamaker[i][j] * M_PI * pow(radius[i][j], 4.0) / 32.0;
	vdw[j][i] = vdw[i][j];

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

double point_to_segment(double* p, double* p1, double* n, double L)
{
	double lambda = (n[0]*(p[0]-p1[0]) + n[1]*(p[1]-p1[1]) + n[2]*(p[2]-p1[2])) / L;
	if (lambda < 0) lambda = 0;
	if (lambda > L) lambda = L;

	double q[3];
	for (int i = 0; i < 3; i++)
		q[i] = p1[i] + lambda * L * n[i] - p[i];
	
	return norm(q);
}

double min(double a, double b)
{
	return (a < b) ? a : b;
}

double g(double x)
{
	double s = (x < 0) ? -1 : 1;
	return 0.5 * s * min(1, 1.5*abs(x));
}

double gamma(double xp, double xm, double yp, double ym)
{
	return min(g(xp) - g(xm), g(yp) - g(ym));
}

double heaviside(double x)
{
	return (x < 0) ? 0 : 1;
}

double sgn(double x)
{
	return (x < 0) ? -1 : 1;
}


double PairTubeTube::calculate_force(
	double* ix, double* jx, double* kx, double* lx,
	int type1, int type2,
	double* fi, double* fj, double* fk, double* fl
	)
{
	for (int i = 0; i < 3; i++)
	{
		fi[i] += 1;
		fj[i] += 1;
		fk[i] += 1;
		fl[i] += 1;
	}

	return 0;
	// vdw[type1][type2]
	
	// Diameter a
	double a = 2*radius[type1][type2];

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
	double L1 = norm(nx);
	double L2 = norm(ny);

	// Single length (arithmetic or geometric)
	double L = (L1 + L2) / 2;

	// Normalize nx and ny
	scale(1/L1, nx);
	scale(1/L2, ny);

	// Calculate nz
	cross(nx, ny, nz);
	scale(sgn(dot(c,nz)), nx);
	
	// Calculate cos_t and sin_t
	double cos_t = dot(nx, ny);
	double sin_t = norm(nz);
	double acos_t = fabs(cos_t);
	double asin_t = fabs(sin_t);

	// Define x0 and y0
	double x0 = 0;
	double y0 = 0;

	// Adjust nz if parallel rods
	// Otherwise calculate x0 and y0
	if (asin_t < 0.05)
	{
		double c_norm = norm(c);
		nz[0] = c[0] / c_norm;
		nz[1] = c[1] / c_norm;
		nz[2] = c[2] / c_norm;
	}
	else
	{
		double cnx = dot(c, nx);
		double cny = dot(c, ny);
		x0 =  (cny*cos_t - cnx) / (sin_t*sin_t);
		y0 = -(cnx*cos_t - cny) / (sin_t*sin_t);
	}

	// Calculate r
	double r = fabs(dot(c, nz));
	if (r < a)
	{
		double abc = point_to_segment(kx, ix, nx, L1);
		double abd = point_to_segment(lx, ix, nx, L1);
		double cda = point_to_segment(ix, kx, ny, L2);
		double cdb = point_to_segment(jx, kx, ny, L2);

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
	}

	/* ---------------------------------------- */
	double F1[3], F2[3], tau1[3], tau2[3];

	double v = acos_t + 3*L*asin_t / (4*(r+a));

	double xp = 4*(v*(x0 + L/2) - y0*cos_t) / (3*L);
	double xm = 4*(v*(x0 - L/2) - y0*cos_t) / (3*L);
	double yp = 4*(v*(y0 + L/2) - x0*cos_t) / (3*L);
	double ym = 4*(v*(y0 - L/2) - x0*cos_t) / (3*L);

	double xap = (x0 + L/2)/a + (L-y0*cos_t)/(2*a*v) - 1.4;
	double xam = (x0 - L/2)/a - (L+y0*cos_t)/(2*a*v) + 1.4;
	double yap = (y0 + L/2)/a + (L-x0*cos_t)/(2*a*v) - 1.4;
	double yam = (y0 - L/2)/a - (L+x0*cos_t)/(2*a*v) + 1.4;

	double r_eff = r - gamma(xap, xam, yap, yam) * a;
	double eta = 1 - (xi[type1][type2] * xi[type1][type2]) / (r_eff * r_eff);
	double uvdw = - vdw[type1][type2] * gamma(xp,xm,yp,ym) / (r_eff * (asin_t + 2.35*sqrt(r*r_eff)/L)*(r+0.12*a)*(r+0.12*a)*(r+0.12*a));

	double lambda = (1 + 1/(2 + L*asin_t/a))/r_eff;
	double sigma = 2*xi[type1][type2]*xi[type1][type2] / (r_eff*r_eff*r_eff);
	double dgadr = 9*L*asin_t*(heaviside(2.0/3.0 - fabs(yap))*(L-2*x0*cos_t) - heaviside(2.0/3.0 - fabs(yam))*(L+2*x0*cos_t)) / (32*a*v*v*(r+a)*(r+a));
	double P = sigma*(1 - a*dgadr) - eta*(lambda + 3/(r+0.12*a));

	if (asin_t < 0.05)
	{
		for (int i = 0; i < 3; i++)
		{
			F1[i] = -uvdw * P * nz[i];
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
		double phi1 = 3*(eta*lambda - sigma) * (heaviside(2.0/3.0 - fabs(yap)) - heaviside(2.0/3.0 - fabs(yam))) / (4*v);
		double phi2 = eta * (heaviside(2.0/3.0 - fabs(yp)) - heaviside(2.0/3.0 - fabs(ym))) / (lambda*L);
		for (int i = 0; i < 3; i++)
		{
			double f = (1+v)*cos_t*nx[i] + (v + cos_t*cos_t)*ny[i];
			F1[i] = uvdw*((phi1+phi2)*f/(sin_t*sin_t) - P*nz[i]);
			F2[i] = -F1[i];
		}

		double dvdt = -sgn(cos_t)*sin_t + sgn(sin_t)*3*L*cos_t / (4*(r+a));
		double dgadt = 3*(heaviside(2.0/3.0-fabs(yap))*(y0*(v*sin_t + dvdt*cos_t) - dvdt*L) - heaviside(2.0/3.0-fabs(yam))*(y0*(v*sin_t + dvdt*cos_t) + dvdt*L)) / (8*a*v*v);
		double factor = uvdw*(sigma*dgadt + eta*cos_t/(sin_t*(1+2.35*sqrt(r*r_eff)/(L*asin_t))));
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
			tau1[i] -= factor * nz[i];
			tau2[i] += factor * nz[i];
		}
	}


	/* ---------------------------------------- */


	// Apply force and torque to atoms
	double f1[3], f2[3];
	cross(tau1, nx, f1);
	cross(tau2, ny, f2);

	double t1 = norm(tau1);
	double t2 = norm(tau2);

	double tn1 = norm(f1);
	double tn2 = norm(f2);

	if (tn1 < 0.005*t1)
	{
		scale(0, f1);
	}
	else
	{
		scale(2*t1/(L1*tn1), f1);
	}

	if (tn2 < 0.005*t2)
	{
		scale(0, f2);
	}
	else
	{
		scale(2*t2/(L2*tn2), f2);
	}

	for (int i = 0; i < 3; i++)
	{
		fi[i] += (F1[i] - f1[i]) / 2;
		fj[i] += (F1[i] + f1[i]) / 2;
		fl[i] += (F2[i] - f2[i]) / 2;
		fk[i] += (F2[i] + f2[i]) / 2;
	}

	return uvdw;
}
