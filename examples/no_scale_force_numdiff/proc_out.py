import numpy as np
from ase.io import read,write
import subprocess
#initial script to evaluate finite differences
def numerical_force(atoms,energies,boxdelta = 0.001,perturb_id=0,component=None,direct_delta = True):
    energy_minus = energies[0]
    energy_plus = energies[2]
    if not direct_delta:
        cell = atoms.get_cell()
    else:
        cell = np.eye(3) 
    num_forces = np.zeros((len(atoms),3))
    for celldim in range(3):
        if component != None:
            if celldim == component:
                delta = boxdelta *cell[celldim]
                num_forces[perturb_id] += (energy_minus - energy_plus) / (2 * delta)
        elif component == None:
            delta = boxdelta *cell[celldim]
            num_forces[perturb_id] += (energy_minus - energy_plus) / (2 * delta)
    return num_forces

def get_es(F):
    no_repeat_es = []
    #subprocess.check_output(grep 'mala force coeffs' outnew.txt | awk '{print $NF}'
    result = subprocess.run("grep 'mala force coeffs' %s" % F + " | awk '{print $NF}'", stdout=subprocess.PIPE,shell=True)
    #result = subprocess.run(['grep',"'mala", "force" "coeffs'", ' %s'%F, ' | ', 'awk','{print $NF}' ], stdout=subprocess.PIPE)
    st = result.stdout.decode('utf-8')
    estrs  = st.split('\n')
    for sti in estrs:
        if sti != '':
            if float(sti) not in no_repeat_es:
                no_repeat_es.append(float(sti))
    return no_repeat_es

def get_fpos(f):
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

perturb_id = 0
fdelta = 0.001
dvs = [-fdelta,0,fdelta]
fsx =  ['dump.2.mala','dump.0.mala','dump.1.mala']
fsy =  ['dump.4.mala','dump.0.mala','dump.3.mala']
fsz =  ['dump.6.mala','dump.0.mala','dump.5.mala']

es = get_es('out_allstep.txt')
es_x = [es[2],es[0],es[1]]
es_y = [es[4],es[0],es[3]]
es_z = [es[6],es[0],es[5]]

poss_x = []
poss_x_dists =[]
frcs_x = []
for f in fsx:
    ps1,fs1 = get_fpos(f)
    frcs_x.append(np.array(fs1[perturb_id]))
    poss_x.append(np.array(ps1[perturb_id]))
    poss_x_dists.append(ps1[perturb_id][0])
poss_x_dists=np.array(poss_x_dists)

poss_y = []
poss_y_dists = []
frcs_y = []
for f in fsy:
    ps1,fs1 = get_fpos(f)
    frcs_y.append(np.array(fs1))
    poss_y.append(np.array(ps1))
    poss_y_dists.append(np.array(ps1[perturb_id][1]))

poss_z = []
poss_z_dists = []
frcs_z = []
for f in fsz:
    ps1,fs1 = get_fpos(f)
    frcs_z.append(np.array(fs1))
    poss_z.append(np.array(ps1))
    poss_z_dists.append(np.array(ps1[perturb_id][2]))
grx_x  = np.gradient(es_x,poss_x_dists)

atoms = read('dump.0.mala')
atoms.symbols = ['Be']*len(atoms)
numforce = numerical_force(atoms,es,boxdelta = fdelta,perturb_id=perturb_id,component = 0,direct_delta=True)
print('numerical force')
print(numforce)
print('\n forces')
print(frcs_x)
numforce_y = numerical_force(atoms,es,boxdelta = fdelta,perturb_id=perturb_id,component = 1,direct_delta=True)
print('numerical force y' )
print(numforce_y)
print('\n forces y')
print(frcs_y)
