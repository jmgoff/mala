#!/bin/bash

valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --log-file=vg_out_0.txt \
        /home/jmgoff/Software/lammps_grid_forces/build/lmp -in in.force_interface &> out.txt &
