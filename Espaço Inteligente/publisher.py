#Cria e publica mensagens

#importa canal e mensagem
from is_wire.core import Channel, Message
from is_msgs.image_pb2 import Image
#Image representa a mensagem

# Connect to the broker
channel = Channel("amqp://guest:guest@localhost:5672")

#criação da mensagem
with open('/home/junia/Dropbox/Junia/Fotos/Dataset/Pesagem - Estreantes 2019/MMP/MMP - 171.jpg','rb') as content_file:
    content = content_file.read()
image = Image()
#Conteúdo da imagem pode ser enviado encorporado na mensagem ou referenciado como recurso externo
#.data representa o conteúdo da imagem como um fluxo de bytes
#.uri para imagem de fonte externa
image.data = content


message = Message(content=image)
# Body is a binary field therefore we need to encode the string
#message.body = "Hello!".encode('latin1')
#.encode('latin1') faz a transformação string to buffer binária
#transformação é necessária para enviar a mensagem


#envia a mensagem
# Broadcast message to anyone interested (subscribed)
channel.publish(message, topic="Test.service")