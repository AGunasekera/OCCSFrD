# read version from installed package
from importlib.metadata import version
__version__ = version("occsfrd")

import occsfrd.ansatz
import occsfrd.interface
import occsfrd.solve
import occsfrd.wick
import occsfrd.reference