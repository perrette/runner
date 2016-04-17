"""Basic interaction with operating system, to submit job, run in terminal, modify directories etc...
"""
from subprocess import Popen, PIPE, check_output
import sys, os, shutil #, datetime

# Tools
# -----
def makedirs(dirname):
    '''
    Make a directory (including sub-directories),
    but first ensuring that path doesn't already exist
    or some other error prevents the creation.
    '''   
    
    try:
        os.makedirs(dirname)
        print     'Directory created: ', dirname
    except OSError:
        if os.path.isdir(dirname):
            print 'Directory already exists: ', dirname
            pass
        else:
            # There was an error on creation, so make sure we know about it
            raise

def command(cmd,input=None):
    '''Execute a command and track the errors and output
       Returns tuple: (output,errors)
    '''
    if input is None:
        proc = Popen(cmd,shell=True,stdout=PIPE,stderr=PIPE)
        out = proc.communicate()
    else:
        proc = Popen(cmd,shell=True,stdout=PIPE,stdin=PIPE,stderr=PIPE)
        out = proc.communicate(input)
    
    return out


def _short(param):
    '''Output short string representation of parameter and value.
       Used for automatic folder name generation.'''
       
    # Store the param value as a string
    # Remove the plus sign in front of exponent
    # Remove directory slashes, periods and trailing .nc from string values
    value = "%s" % (param.value)                
    if "+" in value: value = value.replace('+','')  
    
    if "/" in value: value = value.replace('/','')
    if ".." in value: value = value.replace('..','')
    if ".nc" in value: value = value.replace('.nc','')
    
    # Remove all vowels and underscores from parameter name
    name = param.name
    for letter in ['a','e','i','o','u','A','E','I','O','U','_']:
        name = name[0] + name[1:].replace(letter, '')
    
    return ".".join([name,value])

    
def autofolder(params,outfldr0):
    '''Given a list of parameters,
       generate an appropriate folder name.
    '''
    
    parts = []
    
    for p in params:
        parts.append( _short(p) )
        
    # Join the parts together, combine with the base output dir
    autofldr = '.'.join(parts)
    outfldr  = outfldr0 + autofldr + '/'
    
    return outfldr


def ask_user(msg=None, skip=False):
    if msg is not None: print msg
    try:
        response = raw_input("\n[Enter to proceed] or [ctl-c to exit]"+ " or [s to skip]"*skip)
        print "\n"
    except:
        print "\n"
        sys.exit()
    return response


# Run / Submit Job
# ----------------

def run_background(executable, cmd_args=(), ini_dir='.', out_dir="."):
    " execute in the background "
    print "Running job in background: %s" % (executable)
    print "...initial directory : %s" % (ini_dir)
    print "...output directory : %s" % (out_dir)
    cmd = " ".join(cmd_args) if not isinstance(cmd_args, basestring) else cmd_args
    #print "Storing output in: %s" % (out_dir)
    cmd = "'%s' %s > '%s' &" % (executable, cmd, os.path.join(out_dir,"out.out"))
    if ini_dir != os.path.curdir:
        cmd = "cd '%s' && " % (ini_dir) + cmd  # go to initial directory prior to execution

    print cmd
    code = os.system (cmd)

    return code

def run_foreground(executable, cmd_args=(), ini_dir='.'):
    " execute in terminal, with blocking behaviour "
    cmd = " ".join(cmd_args) if not isinstance(cmd_args, basestring) else cmd_args
    cmd = "%s %s" % (executable, cmd)

    # execute from directory...
    if ini_dir != os.path.curdir:
        cmd = "cd '%s' && " % (ini_dir) + cmd

    print cmd
    code = os.system (cmd)
    return code


# Submit job to supercomputer queue
# ---------------------------------
def jobscript_qsub(executable,cmd_args, outfldr,account,wtime, job_class=None):
    '''Definition of the job script'''
    script = """#!/bin/bash
#$ -V                             # Ensure user enivronment variables are available
#####$ -l nodes=1:ppn=1               # Define as a serial job
#$ -cwd                           # To use the current directory
#$ -m ae                          # Send mail when job is aborted (a), begins (b) and ends (e)
####$ -M robinson@fis.ucm.es         # Send mail to this address
#$ -N rembo_sico                  # (nombre del trabajo)
#$ -o %s/out.out                  # (fichero de salida)   $JOB_ID
#$ -e %s/out.err                  # (fichero de error)
####$ -l walltime=%s:00:00            # Set wall time (hh:mm:ss)
# Run the job
#time ./rembo.x output/default
time ./%s %s 
"""  % (outfldr,outfldr,wtime,executable,outfldr)
    return script

def jobscript_ll(executable, cmd_args, outfldr, account, wtime, job_class="Medium", job_type="serial"):
    # https://docs.loni.org/wiki/LoadLeveler_Command_File_Syntax

    return """#!/bin/ksh
# @ class = {job_class}
# @ group = {account}
# @ job_type = {job_type}
# @ executable = {executable}
# @ output = {outfldr}/log.out
# @ error = {outfldr}/log.err
# @ notification = complete
# @ Wall_clock_limit = {wtime}:00:00
# @ queue
    """.format(**locals())

def jobscript_slurm(executable, cmd_args, outfldr, account, wtime=None, job_class="medium", job_name="runner_job"):
    return """#!/bin/bash

#SBATCH --qos={job_class}
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --output={outfldr}/log-%j.out
#SBATCH --error={outfldr}/log-%j.err

{executable} {cmd_args}
    """.format(**locals())

def get_account(knowngroups=['megarun']):
    " return default username and group for loadleveler submission "
    groups = check_output('groups').strip().split()
    group = groups[0]
    if knowngroups:
        for g in groups:
            if g in knowngroups:
                group = g
                break
    return group

def submit_job(executable, cmd_args, outfldr, system="slurm", account=None, username="", wtime=24, **job_info):

    nm_jobscript = 'job.submit'    # Name of job submit script  
    outfldr += '/'
    username = username or os.environ.get('USER')  # Get the current username

    if system == "loadleveler":
        account = account or get_account()
        job_info.pop("ini_dir", None) # check out option (not supported yet)
        script = jobscript_ll(executable, cmd_args, outfldr, account, wtime, **job_info)
        llq = '/opt/ibmll/LoadL/full/bin/llq'
        llsubmit = '/opt/ibmll/LoadL/full/bin/llsubmit'
        llcancel = '/opt/ibmll/LoadL/full/bin/llcancel'

    elif system == "slurm":
        job_info.pop("ini_dir", None) # check out option (not supported yet)
        llsubmit = "sbatch"  # /p/system/slurm/bin/sbatch
        llq = "squeue" 
        llcancel = "scancel"
        #account = account #or check_output("sacctmgr show assoc where user="+username, shell=True).strip().split()[0]
        account = account or get_account()
        script = jobscript_slurm(executable, cmd_args, outfldr, account, wtime, **job_info)

    elif system == "qsub":
        raise NotImplementedError("Alex, take a look: llq, llsubmit, llcancel need to be defined")
        job_info.pop("ini_dir", None) # check out option (not supported yet)
        account = account or get_account()
        script = jobscript_qsub(executable, cmd_args, outfldr, account, wtime, **job_info)

    else:
        raise NotImplementedError("Unknown cluster queuing system: "+repr(system))

    jobfile1 = open(nm_jobscript,'w').write(script)
    jobfile2 = open(outfldr + nm_jobscript,'w').write(script)
    
    # Copy the job script into output directory for posterity
    if os.path.isfile (outfldr + nm_jobscript): 
        print "Created jobscript file(s): " + nm_jobscript
    
    # Send the submit command to loadleveler
    stat = command("%s %s" % (llsubmit,nm_jobscript))
    print stat[0]
    
    # Check to see if job has actually been submitted
    proc = command("%s -u %s" % (llq,username))
    jobsCheck = proc[0]
    
    if "currently no job" in jobsCheck:
        print "Error in llsubmit: job does not appear in queue!"
        sys.exit(2)

    return stat


