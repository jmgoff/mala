import numpy as np

#Crude script to process forces from mala in lammps dump format:
"""
dump.3.mala:1 1 1.75588 0.321004 2.67767 -0.0129543 -0.0071616 -0.171865
dump.4.mala:1 1 1.75688 0.321004 2.67767 -0.0130768 -0.00710031 -0.171848
dump.5.mala:1 1 1.75788 0.321004 2.67767 -0.0138251 -0.00711612 -0.17288
"""

#NOTE you will need to run the command:
# grep '"energy"' out.txt 
# to parse the energy values from the lammps output. Put them in the 'es' list below
es = [14.752908018811276,14.752419449539179,14.751433689037775,14.750937180900088,14.750436600683987]
#es = []


def get_fpos(f):
    # helper function to get atomic forces from lammps dump file. just supply the name of the file
    fs = []
    ps = []
    with open(f,'r') as readin:
        lines = readin.readlines()
        line_item = [l for l in lines if "ITEM: ATOMS" in l][0]
        atidx = lines.index(line_item)
        for line in lines[atidx+1:]:
            l = line.split()
            fl_l = [float(li) for li in l]
            p = fl_l[2:2+3]
            f = fl_l[2+3:2+3+3]
            ps.append(p)
            fs.append(f)
    return ps,fs

# make a list of all files for numerical differentiation steps
fs = ['dump.0.mala','dump.1.mala','dump.3.mala','dump.4.mala','dump.5.mala']
pxs = []
fxs = []
fys = []
fzs = []
for f in fs:
    ps1,fs1 = get_fpos(f)
    px = ps1[0][0]
    fx = fs1[0][0]
    fy = fs1[0][1]
    fz = fs1[0][2]
    pxs.append(px)
    fxs.append(fx)
    fys.append(fy)
    fzs.append(fz)
import numpy as np
pxs = np.array(pxs) - pxs[0]
print (es,pxs)
gr = np.gradient(es,pxs,edge_order=1)
print(gr)
print(fxs)
