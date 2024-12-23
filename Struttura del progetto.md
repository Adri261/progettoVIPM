### Struttura del progetto

1. Train classificatore stupido su prime 20 immagini ecc... per avere una baseline
	1. CNN-KNN-Bag of words
	2. Transfer learning/Feature extraction
2. Data augmentation su le immagini di base, quelle labeled. Usare la libreria fornita nel corso per idee sull'augmentation
3. Provare se vogliamo a fare la segmentazione per eliminare tutto il background dalle immagini
	1. Provare segmentazione tramite superpixels
4. Sfruttare Dataset unlabeled:
	1. Togliere immagini che non c'entrano. 
	2. Gestire label tramite "noisy labels" e vedere cosa succede
	3. Applicare il classificatore migliore ottenuto ed eseguire classificazione, ri-addestrare il classificatore con le label generate(eventualmente considerare solo le prime k-più confident di ogni classe/le immagini con confidence più alta)
	4. Trovare l'easter egg
	5. Similarity Search tramite clustering: estrarre feature da TUTTE le immagini(labeled e unlabeled) e mettere assieme immagini più simili rispetto a feature, accorpandole poi insieme e dandogli una label.
 	6. CBIR con feedback implicito (dico che sono di classe x le prime k immagini di risulatato ->poi ricerco immagini simili anche a loro).
	7. Self-Supervised learning
6. Dataset Degraded:
	1. Analizzare come è stato degradato, eventualmente creare un algoritmo di elaborazione delle immagini "ad hoc" che sia in grado di pulirlo, così da evitare che il classificatore si occupi della pulizia
	2. Applicare la degradazione come data augmentation, in modo che sia il classificatore ad occuparsi della rimozione del rumore

---

### Idee
- [ ] Per Degraded: dato che il dataset non degraded e degraded contengono le stesse identiche immagini, cambia solo il fatto che siano degraded, allora si potrebbe pensare di allenare un'autoencoder che prende come inputi l'immagine degraded e ha come obiettivo di ricostruire l'immagine non-degraded. Poi se questo metodo funziona si potrebbe mettere l'esecuzione dell'autoencoder come step precedente alla classificazione nella pipeline finale
- [ ] Per lettura immagine: aggiungiamo un random crop invece di un resize delle immagini troppo grandi?

