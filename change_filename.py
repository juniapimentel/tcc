from pathlib import Path
import json
import os

directory_in_str = r'/home/junia/Documents/PG/Dataset_MPI/dataset_mpi/pesagem - estreantes 2019'
pathlist = Path(directory_in_str).glob('**/*.json')
#importo todas as imagens e separo-as em listas

for name in pathlist:
    name = str(name)
    name_json = name.replace("MP - 19 ", "MP - 19 - ")
    if name_json != name:
     os.rename(name, name_json)

