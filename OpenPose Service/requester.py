import sys
import cv2
import json
from pathlib import Path
from argparse import ArgumentParser
from is_msgs.image_pb2 import Image, ObjectAnnotations
from is_wire.core import Channel, Subscription, Message, Logger
from google.protobuf.json_format import MessageToDict


IMAGE_EXTENSION = '.jpg'

parser = ArgumentParser()
parser.add_argument('--broker-uri', type=str, default='amqp://localhost:5672')
parser.add_argument('--base-dir', type=str, required=True)
args = parser.parse_args()

log = Logger(name='Requester')
channel = Channel(args.broker_uri)
subscription = Subscription(channel)

log.info('Connected to {}', args.broker_uri)

image_paths = list(map(str, Path(args.base_dir).glob('**/*{}'.format(IMAGE_EXTENSION))))
log.info("Found {} images with '{}' extension on '{}' folder", len(image_paths), IMAGE_EXTENSION, args.base_dir)

for image_path in image_paths:

  image = cv2.imread(image_path)
  cimage = cv2.imencode(ext='.jpeg', img=image, params=[cv2.IMWRITE_JPEG_QUALITY, 80])
  data = cimage[1].tobytes()
  im = Image(data=data)

  msg = Message(content=im, reply_to=subscription)
  channel.publish(message=msg, topic='SkeletonsDetector.Detect')
  log.info("Image '{}' requested", image_path)

  msg = channel.consume()
  skeletons = msg.unpack(ObjectAnnotations)
  skeletons_dict = MessageToDict(skeletons)

  with open(image_path.replace(IMAGE_EXTENSION, '.json'), 'w') as f:
      json.dump(skeletons_dict, f, indent=2, sort_keys=True)

  log.info("Detection saved")

