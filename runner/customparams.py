"""Custom parameter formats: Alex's SICO & REMBO option files, CLIMBER2 run file
"""
from __future__ import print_function, absolute_import
import warnings
from .parameters import Param, Params  # list-based parameter format

# ===========================
# SICO-REMBO parameter format
# ===========================

class SicoRemboParams(Params):
    """ Parse / Format method specific to Sicopolis-Rembo parameter
    """
    @classmethod
    def parse(cls, string):
        """parse SICO- or REMBO-like parameter file, return a Params instance
        """
        params = cls()
        for i, line in enumerate(string.splitlines()):     

            # skip comments or empty lines
            if line.strip() == "" or line.strip().startswith("#"):
                p = Param(name="", help=line)
            else:
                # parse
                try:
                    p = _parse_param_sico(line)
                except Exception as error:
                    print("Error when parsing SICO-REMBO options file at line {}:\n{}\n{}".format(i, line, error))
                    raise
            params.append( p )
        return params

    def format(self):
        return "\n".join( _format_param_sico(p) for p in self )
    

def _parse_param_sico(line, group=None):
    """ parse a one-liner parameter in Alex' format (SICO and REMBO's option files) 
    and return a Param instance.
    """
    assert len(line) > 41, "Line too short. Expected > 41. Got: {}.".format(len(line))
    assert line[40] in ("=", ":"), "Invalid 40th character. Expected ':' or '='. Got: {}. ".format(line[40])

    part  = line.partition(":")
    units = part[0].strip()
    if ":" in part[2]: 
        sep = ":"
    else:
        sep = "="
    part  = part[2].partition(sep)
    name  = part[0].strip()
    value = part[2].strip()

    if sep == '=': # numerical value
        try:
            value = int(value)
        except:
            value = float(value)
    
    # Also save the initial part of the line for re-writing
    comment = line[:41]
    return Param(name, value, group, units=units, help=comment)


def _format_param_sico(param):
    """ Return one-line string representation SICO-REMBO parameters
    suitable for writing back to ioption file.
    """
    # empty parameters, for formatting
    if not param.name:
        return param.help if param.help is not None else ""
    
    return "{units:^9}:   {name:27}{sep} {value}".format(
        units = param.units or '--',
        name = param.name,
        sep = ":" if isinstance(param.value, basestring) else "=",
        value = param.value)
    # return "{:41} {}".format(param.help,param.value)


# =========================
# CLIMBER2 parameter format
# =========================

class Climber2Params(Params):
    """ Parse / Format method specific to Namelist
    """
    @classmethod
    def parse(cls, string):
        """parse CLIMBER2-like parameter file (returns a Params instance)
        """
        params = cls()
        for i, line in enumerate(string.splitlines()):     
            line = line.strip()
            if line == "":
                continue
            elif line.startswith("="): 
                p = Param(name="", help=line) # empty parameter, help only.
            else:
                try:
                    p = _parse_param_climber2(line)
                except Exception as error:
                    raise RuntimeError("Error when parsing CLIMBER2 parameter file at line {}:\n{}\n{}".format(i, line, error))
            params.append( p )
        return params

    def format(self):
        return "\n".join( _format_param_climber2(p) for p in self )

def _parse_param_climber2(line, group=None):
    """ parse a one-liner parameter in ClIMBER2 format and return a Param instance.
    """
    part  = line.split("|")
    string_value = part[0].strip()
    value = eval(string_value)  # maybe convert to numerical type
    name = part[1].strip()
    comment = "|".join(part[2:])
    return Param(name, value, group, help=comment)

def _format_param_climber2(param):
    " format CLIMBER2 parameter "
    if not param.name:
        assert param.help is not None, "no name nor help ?"
        return param.help
    else:
        return " {:<9}| {:7}| {}".format(repr(param.value), param.name, param.help) 


# =====
# Tests
# =====
def test_options_sico():

    filestring = """
    --   :   DATE                       : 2009-09-10
    --   :   INPATH                     : in_sico
##  ------ Grid related parameters
    --   :   GRID                       =  1
    --   :   X0                         = -800.0
    """
    
    print("Read fake param file: ")
    print(filestring)

    params = SicoRemboParams.parse(filestring)

    print("Result:")
    print(params)

    print()
    print("Formatted:")
    print(params.format())

    assert params.get("DATE").value == "2009-09-10"
    assert params.get("INPATH").value == "in_sico"
    assert params.get("GRID").value == 1, "Got: "+repr(params.get("GRID").value)
    assert params.get("X0").value == -800

    # expected = """
    # --   :   DATE                       : 2009-09-10
    # --   :   INPATH                     : in_sico
    # --   :   GRID                       =  1
    # --   :   X0                         = -800.0"""
    #
    # assert params.format() == expected[1:], "error in formatting SICO"

def test_options_climber2():

    filestring = """
0         | KTIME  | real time          | NO         | YES
1.2       | REAL_VALUE | some fake real value for testing
1700   	  | NYRSR  | strat time (yrBP "-"). For KTIME=0 permanent year
============ Start/restart flag ice-sheet module ===========================   
"INP/restart.dat"  | RESTART_IN   | input restart file
    """

    print("Read fake param file: ")
    print(filestring)

    params = Climber2Params.parse(filestring)

    print("Result:")
    print(params)

    print()
    print("Formatted:")
    print(params.format())
    print()

    assert params.get("KTIME").value == 0, "Got: "+ repr(params.get("KTIME").value)
    assert params.get("REAL_VALUE").value == 1.2
    assert params.get("NYRSR").value == 1700
    assert params.get("RESTART_IN").value == "INP/restart.dat"


if __name__ == "__main__":
    test_options_sico()
    test_options_climber2()
