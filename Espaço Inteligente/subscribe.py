#Cria um canal para se conectar ao broker
#Cria um subscription e subscribe a um tópico de interesse para receber mensagens

from is_wire.core import Channel, Subscription
#importará o canal de comunicação com o broker
#subscription no tópico de interesse

# Connect to the broker
channel = Channel("amqp://guest:guest@localhost:5672")
#nome do protocolo, usuário:senha, @localhost é a url do broker, 5672 é a porta padrão

# Subscribe to the desired topic(s)
#cria um subscription (assinatura)
subscription = Subscription(channel)
#se "inscreve" para receber mensagens acerca do tópico de interesse
subscription.subscribe(topic="Test.service")
# ... subscription.subscribe(topic="Other.Topic")
# Blocks forever waiting for one message from any subscription

message = channel.consume()
#recebe a mensagem
print(message)