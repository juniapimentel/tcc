from is_wire.core import Channel, StatusCode, Status
from is_wire.rpc import ServiceProvider, LogInterceptor
from google.protobuf.struct_pb2 import Struct
from is_msgs.image_pb2 import Image
import time


#Serviço irá retornar o output desejado 


channel = Channel("amqp://guest:guest@localhost:5672")

provider = ServiceProvider(channel)
logging = LogInterceptor()  # Log requests to console
provider.add_interceptor(logging)

provider.delegate(
    topic="CameraGateway.\d+.Frame",
    function=SkeletonsDetector.Detect	,
    request_type=Image,
    reply_type=ObjectAnnotations)
#no serviço para utilizar o openpose, esses parâmetros irão mudar

provider.run() # Blocks forever processing requests