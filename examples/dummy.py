"""dummy model for testing

input: "aa" and "bb" parameters 
    param file in json format, or command line

output "aa" and "bb":
    output.json : output aa and bb
    output : output aa and bb
"""
from __future__ import print_function
import os
import time
import json
import argparse 

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('out')
parser.add_argument('--params-file')
parser.add_argument('--aa', type=float)
parser.add_argument('--bb', type=float)
parser.add_argument('--sleep', type=int)
parser.add_argument('--hang-if-not-aa', action='store_true')

o = parser.parse_args()

# define parameters
aa = 1
bb = 2

# ...from file
if o.params_file:
    params = json.load(open(o.params_file))
    aa = params.pop('aa', aa)
    bb = params.pop('bb', bb)

# ...from command-line
if o.aa is not None: aa = o.aa
if o.bb is not None: bb = o.bb

if o.hang_if_not_aa and not aa:
    o.sleep = 100

if o.sleep:
    print('wait '+str(o.sleep)+' sec')
    time.sleep(o.sleep)

# output variables
output = {'aa':aa,'bb':bb}

print("Model state:", output)

path = os.path.join(o.out, 'output')

print("Write output to", path)
with open(path, 'w') as f:
    for k in output:
        f.write("{} {}\n".format(k, output[k]))

print("Write output to", path+'.json')
with open(path+'.json', 'w') as f:
    json.dump(output, f, sort_keys=True)
