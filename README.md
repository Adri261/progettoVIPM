# (Food) Image CLassification with very few data

## Task: classifiare ogni immagine in una categoria
- 251 classi
- 20 immagini di addestramento per classe
- dati *in the wild* e rumorosi: molte immagini mostrano cibi con la confezione, alcune immagini presentano loghi/watermark
- presenza di un *unlabeled* set contentente dati non etichettati dal quale si può atttingere per fare Content Based Image Retrieval o altro.

### Per più info guarda la presentazione con il quale abbiamo esposto il problema e il nostro approccio:
Presentazione: [pdf](./slides.pdf)

### Cartelle della repository:
- Utils: Contiene script di python generici importati in gran parte dei notebook
- Misc: contiene notebook di python utili per varie esecuzioni
- Stratified K-fold: risultati del training tramite k fold
- BOW: prove con Bag of Words di descrittori
- Plain Augmentation: augmentation dei dati di training con Albumentation
- Semi supervised: prove di Semi Supervised Learning
- SSL-self supervised: prove di Self Supervised Learning

