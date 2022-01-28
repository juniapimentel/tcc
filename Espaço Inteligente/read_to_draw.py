from pathlib import Path
directory_in_str = r'/home/junia/Dropbox/Junia/Fotos/Dataset'
pathlist = Path(directory_in_str).glob('**/*.jpg')
#importo todas as imagens e separo-as em listas

for name in pathlist:
    name = str(name)
    name_json = name.replace(".jpg", ".json")
    output_image = name.replace(".jpg", " - output.jpg")
    print(output_image)
