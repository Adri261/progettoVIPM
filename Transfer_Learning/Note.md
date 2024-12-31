### Note su transfer learning

Dato che tutte le reti principali sono presenti come pretrainate su PyTorch userò quello invece di Keras come Framework

Tutti i dati sono caricati su memoria GPU quindi è fondamentale che funzioni cuda 

Dopo aver allenato e testato varie cose mi occuperò di trasferire il codice su matlab eventualmente

Accuracy transfer learning alexnet:
- ultimo strato: quello scritto in chat
- ultimi 2 strati:  10.3885275971319 %
- ultimi 3 strati: 10.196765049191263 %
