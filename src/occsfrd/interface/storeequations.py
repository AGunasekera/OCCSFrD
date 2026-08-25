import pickle
import sys
import requests

# def saveEquation(equation):
#     d = {}
#     with open("unlinkedNormalOrderedCCSDDoubletEquations.pkl", 'wb') as f:
#         p = pickle.Pickler(f)
#         p.dump(d)

# def loadEquation(equation):
#     with open("CCDEquations.pkl", 'rb') as f:
#         up = pickle.Unpickler(f)
#         d = up.load()

def save(name, equations, tensors, specificIndices=[]):
    '''
    Save equations and tensors as pickle file

    Inputs:
    name             (str): name of the pickle file into which the objects are saved
    equations       (list): list of the energy and amplitude equations as tensor.TensorSum objects
    tensors         (list): list of the fock, 2-body interaction, and amplitude tensors as tensor.Tensor objects
    specificIndices (list): list of any specificIndex objects appearing in the equations

    Returns:

    Side-effects:
    [name].pkl contains a pickled dictionary with the equations (keyed with "equations") and tensors (keyed with "tensors")
    '''
    with open(name + ".pkl", 'wb') as f:
        d = {"equations": equations, "tensors": tensors, "specificIndices": specificIndices}
        p = pickle.Pickler(f)
        p.dump(d)

def load(name):
    '''
    Load equations and tensors from pickle file
    
    Inputs:
    name (str): [name].pkl is the file containing the pickled equation and tensor objects
    
    Returns:
    (dict): dictionary containing the equations (keyed with "equations") and tensors (keyed with "tensors") that were stored in [name].pkl
    '''
    with open(name + ".pkl", 'rb') as f:
        up = pickle.Unpickler(f)
        return up.load()

def downloadFromZenodo(eqnName, ACCESS_TOKEN=""):
    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}
    deposition_url = 'https://sandbox.zenodo.org/api/deposit/depositions'
    which_deposition = 0
    r = requests.get(deposition_url, headers=headers)
    r = requests.get("%s/%s" % (deposition_url, r.json()[which_deposition]['id']), headers=headers)
    fNamesAndDownloadLinks = [(file['filename'], file['links']['download']) for file in r.json()['files']]

    eqnFileName = eqnName + ".eqn"

    for name, link in fNamesAndDownloadLinks:
        if name == eqnFileName:
            with open(name[:-3]+"pkl", 'wb') as f:
                r = requests.get(link, headers=headers)
                f.write(r.content)

    sys.modules['openshellcc'] = sys.modules['occsfrd']
    return load(eqnName)