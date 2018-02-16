import numpy as np
import sys
import os
import subprocess

import nipype.interfaces.ants as ants
print(os.getcwd())
import BCBlib.tools.constants as cst

def ctrl(opt_name, opt_value):
    if opt_value == None:
        return ""
    else:
        return " " + opt_name + " " + str(opt_value)

def skull_strip(dim, anatomical, brain_proba, template):
    be = ants.BrainExtraction()
    be.inputs.dimension = dim
    be.inputs.anatomical_image = anatomical
    be.inputs.brain_template = brain_proba
    be.inputs.brain_probability_mask = template
    be.cmdline

    # We can use the mask of the brain resulting from BrainExtraction
    # in Atropos to define the regions we want to segment.


def segment(dim, anatomical, mask=None, nb_class=None,
            out_pref=None, priors=None):
    opt = "-d " + str(dim) \
        + " -a " + anatomical \
        + ctrl("-x", mask) \
        + ctrl("-c", nb_class) \
        + ctrl("-o", out_pref) \
        + ctrl("-p", priors)
    s = subprocess.Popen(["antsAtroposN4.sh", opt], stdout=subprocess.PIPE)
    out, err = s.communicate()
    if err != None:
        print("Error in antsAtroposN4.sh : \n" + err)
    else:
        print(out)

dim = sys.argv[1]
anat = sys.argv[2]
# opt
mask = sys.argv[3]
nb_class = sys.argv[4]
out_pref = sys.argv[5]
priors = sys.argv[6]
