from pathlib import Path
directory_in_str = r'/home/junia/Dropbox/Junia/Fotos/Dataset'
pathlist = Path(directory_in_str).glob('**/*.jpg')
jsonlist = Path(directory_in_str).glob('**/*.json')
#importo todas as imagens e separo-as em listas
mp = []
mmp = []
for name in pathlist:
    name = str(name)
    if '/MP/' in name:
        mp.append(name)
    elif '/MMP/' in name:
        mmp.append(name)
#Cria label de cada imagem separadamente
label_mp = list(['MP']*len(mp))
dic_mp = dict(zip(mp,label_mp))
label_mmp = list(['MMP']*len(mmp))
dic_mmp = dict(zip(mmp,label_mmp))

