project('occsfrd', 'c', version: '0.1.4.a')

add_languages('fortran', native: false)

py = import('python').find_installation(pure: false)

# Install the Python package files
py.install_sources(
  'src/occsfrd/__init__.py',
  subdir: 'occsfrd',
)

py.install_sources(
  'src/occsfrd/ansatz/__init__.py',
  'src/occsfrd/ansatz/closedshellcc.py',
  'src/occsfrd/ansatz/normalorderedcc.py',
  'src/occsfrd/ansatz/utils.py',
  subdir: 'occsfrd/ansatz',
)

py.install_sources(
  'src/occsfrd/interface/__init__.py',
  'src/occsfrd/interface/pyscfoperator.py',
  'src/occsfrd/interface/storeequations.py',
  'src/occsfrd/interface/texify.py',
  subdir: 'occsfrd/interface',
)

py.install_sources(
  'src/occsfrd/reference/__init__.py',
  'src/occsfrd/reference/UGA.py',
  subdir: 'occsfrd/reference',
)

py.install_sources(
  'src/occsfrd/solve/__init__.py',
  'src/occsfrd/solve/cc.py',
  'src/occsfrd/solve/diis.py',
  subdir: 'occsfrd/solve',
)

py.install_sources(
  'src/occsfrd/wick/__init__.py',
  'src/occsfrd/wick/contractions.py',
  'src/occsfrd/wick/index.py',
  'src/occsfrd/wick/operator.py',
  'src/occsfrd/wick/tensor.py',
  subdir: 'occsfrd/wick',
)

py.install_sources(
  'src/occsfrd/wick/contract/__init__.py',
  subdir: 'occsfrd/wick/contract',
)

# NumPy >=2.0 provides include dirs via dependency()
np_dep = dependency('numpy')

incdir_f2py = run_command(py,
#  ['-c', 'print("print")'],
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check : true
).stdout().strip()

incdir = include_directories('src/occsfrd/wick/contract/')
#incdir_contract = run_command(py,
#  ['-c', 'print("print")'],
#  ['-c', 'import occsfrd.wick.contract'],
#  check : true
#).stdout().strip()

# f2py include dir (for fortranobject.h) is not in dependency('numpy'),
# so add it separately
f2py_dep = declare_dependency(
  include_directories : incdir_f2py,
)

# Generate the f2py wrappers
contract_source = custom_target('contractEXT',
  build_by_default: true,
  input : ['src/occsfrd/wick/contract/contract.f90'],
#  output : ['contractEXT.so', 'contractSubroutine.pyf', 'contract.x'],
  output : ['contractmodule.c'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'contract', '--lower']
)

#py.extension_module(
#    'occsfrd.wick.contract.contract.f90',
#    'occsfrd/wick/contract/contractSubroutine.pyf',
#    install: true,
#)

py.extension_module('contractEXT',
#  ['contract.f90', 'contractEXT.so', incdir_f2py / 'contract.x'],
#  [incdir],
  [contract_source],
#  include_directories : 'src/occsfrd/wick/contract/',
  dependencies : [np_dep, f2py_dep],
  subdir: 'occsfrd/wick/contract/',
  install : true,
)