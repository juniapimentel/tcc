from PIL import Image
from pathlib import Path
import os

size = (224,224)
input_directory ='/home/junia/Documents/PG/Dataset_Cropped_200/MP' 
out_directory = '/home/junia/Documents/PG/augmentor/dataset/train/MP' 
 
os.makedirs(out_directory, exist_ok=True)

files = Path(input_directory).glob("**/*.jpg")

for filename in files:
  # assuming 'import from PIL *' is preceding
  thumbnail = Image.open(filename)
  # generating the thumbnail from given size
  thumbnail.thumbnail(size, Image.ANTIALIAS)

  offset_x = max((size[0] - thumbnail.size[0]) // 2, 0)
  offset_y = max((size[1] - thumbnail.size[1]) // 2, 0)
  offset_tuple = (offset_x, offset_y) #pack x and y into a tuple

  # create the image object to be the final product
  final_thumb = Image.new(mode='RGB',size=size,color=(0,0,0))
  # paste the thumbnail into the full sized image
  final_thumb.paste(thumbnail, offset_tuple)
  # save (the PNG format will retain the alpha band unlike JPEG)

  # final_thumb.show()
  final_thumb.save(str(filename).replace(input_directory, out_directory))

