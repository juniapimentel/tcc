import Augmentor

#espelhamento

def createAugmentationPipeline():
  p = Augmentor.Pipeline()
  p.flip_left_right(probability=0.5)
  p.flip_top_bottom(probability=0.5)
  p.rotate90(probability=0.5)
  p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
  p.random_erasing(probability=0.5, rectangle_area=0.2)
  p.random_brightness(probability=0.2, min_factor=0.3, max_factor=1)
  return p

