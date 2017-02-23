from __future__ import print_function
import unittest
import os, shutil
import six
import json
import logging
from subprocess import check_call

JOB = "./scripts/job"

if six.PY2:
    from subprocess import check_output
    def getoutput(cmd):
        return check_output(cmd, shell=True)
else:
    from subprocess import getoutput


class TestSample(unittest.TestCase):

    def test_product(self):
        out = getoutput(JOB+' product a=2,3,4 b=0,1')
        self.assertEqual(out.strip(), """
     a      b
     2      0
     2      1
     3      0
     3      1
     4      0
     4      1
                         """.strip())

    def test_sample(self):
        out = getoutput(JOB+' sample a=U?0,1 b=N?0,1 --size 10 --seed 4')
        if six.PY3:
            # note: python3 uses more digits
            self.assertEqual(out.strip(),"""
a      b
0.4252982362383444 0.9889538055947533
0.90441600579315 2.6248228301550833
0.6862993235599223 0.7054452199344784
0.3976274454776242 -0.766770633921025
0.5779382921793753 -0.5226094671315683
0.09670298390136767 -0.1421540745795235
0.71638422414047 0.04957259589653996
0.2697728824597271 0.5196323235536475
0.19726843599648844 -1.6006861519796032
0.8008986097667555 -0.9483266285993096

                             """.strip())
        else:
            self.assertEqual(out.strip(),"""
     a      b
0.425298236238 0.988953805595
0.904416005793 2.62482283016
0.68629932356 0.705445219934
0.397627445478 -0.766770633921
0.577938292179 -0.522609467132
0.0967029839014 -0.14215407458
0.71638422414 0.0495725958965
0.26977288246 0.519632323554
0.197268435996 -1.60068615198
0.800898609767 -0.948326628599
                             """.strip())

class TestRunBase(unittest.TestCase):

    def setUp(self):
        if os.path.exists('out'):
            raise RuntimeError('remove output directory `out` before running run tests')

    def tearDown(self):
        if os.path.exists('out'):
            shutil.rmtree('out') # clean up after each individual test


class TestParamsIO(TestRunBase):


    def test_paramsio_args(self):
        out = getoutput(JOB+' run -p a=2,3,4 b=0,1 -o out --shell -- echo --a {a} --b {b} --out {}')
        self.assertEqual(out.strip(),"""
--a 2 --b 0 --out out/0
--a 2 --b 1 --out out/1
--a 3 --b 0 --out out/2
--a 3 --b 1 --out out/3
--a 4 --b 0 --out out/4
--a 4 --b 1 --out out/5
                         """.strip())

    def test_paramsio_args_prefix(self):
        out = getoutput(JOB+' run -p a=2,3,4 b=0,1 -o out --shell --arg-prefix "--{} " --arg-out-prefix "--out " -- echo')
        self.assertEqual(out.strip(),"""
--out out/0 --a 2 --b 0
--out out/1 --a 2 --b 1
--out out/2 --a 3 --b 0
--out out/3 --a 3 --b 1
--out out/4 --a 4 --b 0
--out out/5 --a 4 --b 1
                         """.strip())

    def test_paramsio_env(self):
        out = getoutput(JOB+' run -p a=2,3 b=0. -o out --shell --env-prefix "" -- bash examples/dummy.sh')
        self.assertEqual(out.strip(),"""
RUNDIR out/0
a 2
b 0.0
RUNDIR out/1
a 3
b 0.0
                         """.strip())

    def test_paramsio_file_linesep(self):
        out = getoutput(JOB+' run -p a=2,3,4 b=0,1 -o out --file-name params.txt --file-type linesep --line-sep " " --shell cat {}/params.txt')
        self.assertEqual(out.strip(),self.linesep.strip())

    linesep = """
a 2
b 0
a 2
b 1
a 3
b 0
a 3
b 1
a 4
b 0
a 4
b 1
    """

    def test_paramsio_file_linesep_auto(self):
        out = getoutput(JOB+' run -p a=2,3,4 b=0,1 -o out --file-name params.txt --shell cat {}/params.txt')
        self.assertEqual(out.strip(),self.linesep.strip())

    def test_paramsio_file_namelist(self):
        out = getoutput(JOB+' run -p g1.a=0,1 g2.b=2. -o out --file-name params.txt --file-type namelist --shell  cat {}/params.txt')
        self.assertEqual(out.strip(), self.namelist.strip())
        
    namelist = """
&g1
 a               = 0          
/
&g2
 b               = 2.0        
/
&g1
 a               = 1          
/
&g2
 b               = 2.0        
/
    """

    def test_paramsio_file_namelist_auto(self):
        out = getoutput(JOB+' run -p g1.a=0,1 g2.b=2. -o out --file-name params.nml --shell  cat {}/params.nml')
        self.assertEqual(out.strip(), self.namelist.strip())


    def test_paramsio_file_json(self):
        getoutput(JOB+' run -p a=2 b=0,1 -o out --file-name params.json --file-out params.json echo')
        self.assertEqual(json.load(open('out/0/runner.json'))['output'], {'a':2,'b':0})
        self.assertEqual(json.load(open('out/1/runner.json'))['output'], {'a':2,'b':1})

    def test_custom(self):
        getoutput(JOB+' run -p a=2 b=0,1 -m examples/custom.py -o out --file-name params.json')
        self.assertEqual(json.load(open('out/0/runner.json'))['output'], {'a':2,'b':0})
        self.assertEqual(json.load(open('out/1/runner.json'))['output'], {'a':2,'b':1})


class TestRunSubmit(TestRunBase):

    def test_shell(self):
        out = getoutput(JOB+' run -p a=2,3,4 b=0,1 -o out --shell -- echo --a {a} --b {b} --out {}')
        self.assertEqual(out.strip(),"""
--a 2 --b 0 --out out/0
--a 2 --b 1 --out out/1
--a 3 --b 0 --out out/2
--a 3 --b 1 --out out/3
--a 4 --b 0 --out out/4
--a 4 --b 1 --out out/5
                         """.strip())

    def test_main(self):
        _ = getoutput(JOB+' run -p a=2,3,4 b=0,1 -o out -- echo --a {a} --b {b} --out {}')
        out = getoutput('cat out/*/log.out')
        self.assertEqual(out.strip(),"""
--a 2 --b 0 --out out/0
--a 2 --b 1 --out out/1
--a 3 --b 0 --out out/2
--a 3 --b 1 --out out/3
--a 4 --b 0 --out out/4
--a 4 --b 1 --out out/5
                         """.strip())


class TestRunIndices(TestRunBase):

    def test_shell(self):
        out = getoutput(JOB+' run -p a=2,3,4 b=0,1 -o out --shell -j 0,2-4 -- echo --a {a} --b {b} --out {}')
        self.assertEqual(out.strip(),"""
--a 2 --b 0 --out out/0
--a 3 --b 0 --out out/2
--a 3 --b 1 --out out/3
--a 4 --b 0 --out out/4
                         """.strip())


class TestAnalyze(unittest.TestCase):

    fileout = 'output.json'

    @classmethod
    def setUpClass(cls):
        if os.path.exists('out'):
            raise RuntimeError('remove output directory `out` before running tests')
        cmd = (JOB+' run -p a=1,2 b=0. -o out'
                           +' --file-out '+cls.fileout
                           +' --shell python examples/dummy.py {} --aa {a} --bb {b}')
        print(cmd)
        check_call(cmd, shell=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('out'):
            shutil.rmtree('out') # clean up after each individual test

    def test_state(self):
        check_call(JOB+' analyze out -v aa bb', shell=True)
        out = open('out/output.txt').read()
        self.assertEqual(out.strip(),"""
	aa     bb
   1.0    0.0
   2.0    0.0
                         """.strip())

    def test_state_mixed(self):
        check_call(JOB+' analyze out -v aa -l bb=N?0,1', shell=True)
        out = open('out/output.txt').read()
        self.assertEqual(out.strip(),"""
	aa     bb
   1.0    0.0
   2.0    0.0
                         """.strip())

    def test_like(self):
        check_call(JOB+' analyze out -l aa=N?0,1', shell=True)
        out = open('out/loglik.txt').read()
        self.assertEqual(out.strip(),"""
-1.418938533204672670e+00
-2.918938533204672670e+00
                         """.strip())

class TestAnalyzeLineSep(TestAnalyze):
    fileout = 'output'



if __name__ == '__main__':
    unittest.main()
