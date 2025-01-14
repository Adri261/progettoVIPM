### Note su transfer learning

Dato che tutte le reti principali sono presenti come pretrainate su PyTorch userò quello invece di Keras come Framework

Tutti i dati sono caricati su memoria GPU quindi è fondamentale che funzioni cuda 

Dopo aver allenato e testato varie cose mi occuperò di trasferire il codice su matlab eventualmente

I prossimi risultati sono ottenuti da training con 20 epoche e batch size 128

Accuracy transfer learning alexnet: (bgr)
- ultimo strato: quello scritto in chat
- ultimi 2 strati:  10.3885275971319 %
- ultimi 3 strati: 10.196765049191263 %

Accuracy rgb:
- ultimo strato: 12.739703184925796 %
- ultimi 2 strati: 13.873603468400866
- ultimi 3 strati: 13.765215941303985 %

Dopo aver allenato e testato varie cose mi occuperò di trasferire il codice su matlab eventualmente

I prossimi risultati sono ottenuti da training con 50 epoche e batch size 128, dataset correttamente caricato con immagini RGB

- ultimo strato: 14.24045356011339 %
- ultimi 2 strati: 14.957478739369684 %
- ultimi 3 strati: 14.507253626813407 %

Dato che i grafici di training-val loss non divergono troppo per ultimo strato e ultimi 2 strati ho provato ad aumentare 
il numero totale di epoche a 100, Risultati:

- ultimo strato: 14.298816074704018 %
- ultimi 2 strati: 14.473903618475906 %

