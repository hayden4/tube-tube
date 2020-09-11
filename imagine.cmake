# CMAKE file for LAMMPS packages available on imagine


set(ALL_PACKAGES CLASS2 COLLOID MC MISC RIGID OPT MOLECULE)

foreach(PKG ${ALL_PACKAGES})
  set(PKG_${PKG} ON CACHE BOOL "" FORCE)
endforeach()
