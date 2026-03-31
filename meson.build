project('occsfrd', 'c')

add_languages('fortran', native: false)

py = import('python').find_installation(pure: false)

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

py.extension_module('occsfrd.wick.contract.contractEXT',
#  ['contract.f90', 'contractEXT.so', incdir_f2py / 'contract.x'],
#  [incdir],
  [contract_source],
#  include_directories : 'src/occsfrd/wick/contract/',
  dependencies : [np_dep, f2py_dep],
  subdir: 'src/occsfrd/wick/contract/',
  install : true,
)

# Install the Python package files
py.install_sources(
  'src/occsfrd/__init__.py',
  subdir: 'occsfrd',
)