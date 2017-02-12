import unittest
import os, shutil
from subprocess import check_output, call

JOB = "./scripts/job --debug"

class TestSample(unittest.TestCase):

    def test_product(self):
        out = check_output(JOB+' product a=2,3,4 b=0,1', shell=True)
        self.assertEqual(out.strip(),"""
     a      b
     2      0
     2      1
     3      0
     3      1
     4      0
     4      1
                         """.strip())


    def test_sample(self):
        out = check_output(JOB+' sample a=U?0,1 b=N?0,1 --size 10 --seed 4', shell=True)
        # FIXME: python3 uses more digits, how to make a test that works for both versions?
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


class TestRun(unittest.TestCase):

    def setUp(self):
        if os.path.exists('out'):
            raise RuntimeError('remove output directory `out` before running run tests')

    def tearDown(self):
        if os.path.exists('out'):
            shutil.rmtree('out') # clean up after each individual test

    def test_paramsio_args(self):
        out = check_output(JOB+' run -p a=2,3,4 b=0,1 -o out --shell -- echo --a {a} --b {b} --out {}', shell=True)
        self.assertEqual(out.strip(),"""
--a 2 --b 0 --out out/0
--a 2 --b 1 --out out/1
--a 3 --b 0 --out out/2
--a 3 --b 1 --out out/3
--a 4 --b 0 --out out/4
--a 4 --b 1 --out out/5
                         """.strip())

    def test_paramsio_args_prefix(self):
        out = check_output(JOB+' run -p a=2,3,4 b=0,1 -o out --shell --arg-prefix "--{} " --arg-out-prefix "--out " -- echo', shell=True)
        self.assertEqual(out.strip(),"""
--out out/0 --a 2 --b 0
--out out/1 --a 2 --b 1
--out out/2 --a 3 --b 0
--out out/3 --a 3 --b 1
--out out/4 --a 4 --b 0
--out out/5 --a 4 --b 1
                         """.strip())

    def test_paramsio_env(self):
        out = check_output(JOB+' run -p a=2,3 b=0. -o out --shell --env-prefix "" -- bash scripts/dummy.sh', shell=True)
        self.assertEqual(out.strip(),"""
RUNDIR out/0
a 2
b 0.0
RUNDIR out/1
a 3
b 0.0
                         """.strip())

    def test_paramsio_file_linesep(self):
        out = check_output(JOB+' run -p a=2,3,4 b=0,1 -o out --file-name params.txt --file-type linesep --line-sep " " --shell cat {}/params.txt', shell=True)
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
        out = check_output(JOB+' run -p a=2,3,4 b=0,1 -o out --file-name params.txt --shell cat {}/params.txt', shell=True)
        self.assertEqual(out.strip(),self.linesep.strip())

    def test_paramsio_file_namelist(self):
        out = check_output(JOB+' run -p g1.a=0,1 g2.b=2. -o out --file-name params.txt --file-type namelist --shell  cat {}/params.txt', shell=True)
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
        out = check_output(JOB+' run -p g1.a=0,1 g2.b=2. -o out --file-name params.nml --shell  cat {}/params.nml', shell=True)
        self.assertEqual(out.strip(), self.namelist.strip())


class TestAnalyze(unittest.TestCase):

    fileout = 'output.json'

    @classmethod
    def setUpClass(cls):
        if os.path.exists('out'):
            raise RuntimeError('remove output directory `out` before running tests')
        out = check_output(JOB+' run -p a=1,2 b=0. -o out'
                           +' --file-out '+cls.fileout
                           +' --shell python scripts/dummy.py {} --aa {a} --bb {b}', shell=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('out'):
            shutil.rmtree('out') # clean up after each individual test

    def test_state(self):
        assert call(JOB+' analyze out -v aa bb', shell=True) == 0
        out = open('out/state.txt').read()
        self.assertEqual(out.strip(),"""
	aa     bb
   1.0    0.0
   2.0    0.0
                         """.strip())

    def test_state_mixed(self):
        assert call(JOB+' analyze out -v aa -l bb=N?0,1', shell=True) == 0
        out = open('out/state.txt').read()
        self.assertEqual(out.strip(),"""
	aa     bb
   1.0    0.0
   2.0    0.0
                         """.strip())

    def test_like(self):
        assert call(JOB+' analyze out -l aa=N?0,1', shell=True) == 0
        out = open('out/loglik.txt').read()
        self.assertEqual(out.strip(),"""
-1.418938533204672670e+00
-2.918938533204672670e+00
                         """.strip())

class TestAnalyzeLineSep(TestAnalyze):
    fileout = 'output'



if __name__ == '__main__':
    unittest.main()
