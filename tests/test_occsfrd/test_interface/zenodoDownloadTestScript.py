import sys
import requests

assert (len(sys.argv) == 2) and isinstance(sys.argv[1], str)

ACCESS_TOKEN = sys.argv[1]
headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}
deposition_url = 'https://sandbox.zenodo.org/api/deposit/depositions'
which_deposition = 0
r = requests.get(deposition_url, headers=headers)
r = requests.get("%s/%s" % (deposition_url, r.json()[which_deposition]['id']), headers=headers)
fNamesAndDownloadLinks = [(file['filename'], file['links']['download']) for file in r.json()['files']]

eqnFileName = "N0S0_lCCSD.eqn"

for name, link in fNamesAndDownloadLinks:
    if name == eqnFileName:
        with open(name, 'wb') as f:
            r = requests.get(link, headers=headers)
            f.write(r.content)
# r = requests.get(f0download, headers=headers)
# with open(f0name, 'wb') as f0:
#     f0.write(r.content)
# for file in r.json()['files']:
#     filename = file['filename']
#     with open(filename, wb) as f:
#     r = requests.get(r.json()['files'][1]['links']['download'], headers=headers)
#     f.write(r.content)