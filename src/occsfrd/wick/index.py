"""
Index class for implementing Wick contractions
"""

class Index:
    """
    Class for an orbital index

    name             (str) : name given to index
    occupiedInVaccum (bool): whether the index refers to orbitals that are occupied in the vacuum
                             (i.e. hole orbitals) or not (particle orbitals)
    active           (bool): whether the index refers to active (singly-occupied) orbitals or not
    """
    def __init__(self, name, occupiedInVacuum, active=False):
        self.name = name
        self.occupiedInVacuum = occupiedInVacuum
        self.active = active
        self.tuple = (self.name, self.occupiedInVacuum, self.active) # Hashable type

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        return self.tuple == other

    def __str__(self):
        return self.name

class SpecificOrbitalIndex(Index):
    """
    Index subclass for particular orbital index values to be specified in the Ansatz
    (e.g. open-shell orbitals in reference)

    contractedFrom (Index): general index for which this specific index represents a particular value
    """
    def __init__(self, name, occupiedInVacuum=False, active=True, contractedFrom=None, specificIndexValue=None):
        super(SpecificOrbitalIndex, self).__init__(name, occupiedInVacuum, active)
        self.contractedFrom = contractedFrom
        if self.contractedFrom is None:
            self.contractedFrom = self
        self.value = specificIndexValue

    def contractedCopy(self, contractedFrom):
        """
        When contracting a specific index value, keep track of what general index it has contracted with

        contractedFrom (Index): general index with which this index is contracting
        """
        return SpecificOrbitalIndex(self.name, self.occupiedInVacuum, self.active, contractedFrom)

    def __copy__(self):
        return SpecificOrbitalIndex(self.name, self.occupiedInVacuum, self.active, self.contractedFrom)