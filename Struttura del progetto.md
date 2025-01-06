### Struttura del progetto

1. Addestro classificatore stupido su prime 20 immagini ecc... per avere una *baseline*
	1. CNN-KNN-Bag of words
	2. Transfer learning/Feature extraction

2. Data *augmentation* su le immagini di base, quelle labeled. Usare la libreria fornita nel corso per idee sull'augmentation
 
3. Provare se vogliamo a fare la *segmentazione* per eliminare tutto il background dalle immagini
	1. Provare segmentazione tramite superpixels

4. Sfruttare Dataset unlabeled: (nota: trattare (successivamente) le label delle immagini raccattate da questo split come "noisy labels")
	1. Togliere immagini che non c'entrano (*pre-pulizia* di questo split per quello che ci farò con esso).
	2. *CBIR* con feedback implicito: applicare il classificatore migliore ottenuto ed eseguire classificazione, ri-addestrare il classificatore con le 20*251 + le k-più confident di ogni classe/le immagini con confidence più alta di una soglia (le immagini aggunte allo split di train assumo siano di quella classe a questo punto).
	3. Similarity Search tramite clustering: estrarre feature da TUTTE le immagini(labeled e unlabeled) e mettere assieme immagini più simili rispetto a feature, accorpandole poi insieme e dandogli una label.
 	4. *Self-Supervised learning*

4-post. Data *augmentation* su le immagini di base + quelle raccattate dall'unlabeled. 

5. Dataset Degraded:
	1. Analizzare come è stato degradato, eventualmente creare un algoritmo di elaborazione delle immagini "ad hoc" che sia in grado di *pulirlo*, così da evitare che il classificatore si occupi della pulizia
	2. Applicare la *degradazione come data augmentation*, in modo che sia il classificatore ad occuparsi della rimozione del rumore

---

### Idee
- [ ] Per Degraded: dato che il dataset non degraded e degraded contengono le stesse identiche immagini, cambia solo il fatto che siano degraded, allora si potrebbe pensare di allenare un'autoencoder che prende come inputi l'immagine degraded e ha come obiettivo di ricostruire l'immagine non-degraded. Poi se questo metodo funziona si potrebbe mettere l'esecuzione dell'autoencoder come step precedente alla classificazione nella pipeline finale
- [ ] Per lettura immagine: aggiungiamo un random crop invece di un resize delle immagini troppo grandi?


