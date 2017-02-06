from runner.register import jobs, register_job as register

# register job tasks
import runner.ext.namelist
import runner.job.stats  
import runner.job.run
import runner.job.analysis
from runner.job.__main__ import main
