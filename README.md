LAMMPS tube-tube module

All source code located in the src/ directory. To integrate the module into your LAMMPS
distribution, move the files from the src/ directory into the LAMMPS src/ directory. To integrate
the atom.cpp and atom.h files correctly, perform a diff and add the relevant ghost\_atom arrays.

A sample CMake file was added in the top level directory, which was used for building LAMMPS for
testing. Last tested build on Aug 17th, 2019 build of LAMMPS, but tests did not fully pass.
