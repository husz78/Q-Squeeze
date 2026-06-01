\### O co chodzi?



Badają jak najefektywniej zmnieszyć rozmiar LLMa poprzez pruning i knowledge distillation.

Prezentują pewne best practices, które powinno wg nich stosować.



\### Metodologia (pruningu):

Jako, że metody z gradientem potrzebują dużo pamięci i compute to stosują ranking na podstawie aktywacji.

Używają do tego small calibration datasetu (1024 samples) i tylko używając forward pass.



Będziemy liczyli ranking rzeczy na różnych płaszczyznach:

* Depth (warstwy)
* Neurony
* Attention Heads
* Embedding channel

Korzystają z jakichś wzorków na activation-based importance dla Heads, neuronów, embedding channels,

agregując metryki po batchu i seq\_len.





W depth pruning liczą Block Importance oraz ranking po Perplexity:

* BI to cosine similarity na inpucie danej warstwy i jej outpucie. Może być obliczone w jednym forward passie
* W PPL-ranking usuwają warstwę i liczą zmianę w Perplexity dla tego modelu. Nie może być obliczone w jednym forward passie



Robią Iterative Importance: alternują pomiędzy pruningiem i importance estimation w pętli parę razy.

Prunujemy wszystko wsm: Heads, embedding dim, neurony w MLP, i warstwy.
Autorzy ewaluują co jest najlepsze.

Sprytny Trick:
Przy prunowaniu attention head o indeksie k wybierają inny head o indeksie i.
Usuwają całkowicie head\_k i robią head\_i = head\_i + (head\_i - head\_k), więc starają się zachować część informacji

z usuniętych headów.


### Jak wybierają najlepszy model:

Robią architecture search (mając fixed budget i określony search space). Używają importance analysis do tego.
Później dostają ok. 20 kandydatów, którzy są obiecujący i ci kandydaci przechodzą Lightweight Retraining (RT)
(1.8B tokens)

Stosują Knowledge Distillation na teacher (model przed pruningiem) i student (kandydat po pruning) i
używają KL Divergence i kombinacji innych lossów.

Na koniec najlepszy kandydat jeszcze jest Fully REtrained na większym zbiorze danych.

### Wyniki:


Pruning Results i insights:
1. To train a family of LLMs, train the largest one and prune+distill iteratively to smaller LLMs.

2\. Use (batch=L2, seq=mean) importance estimation for width axes and PPL/BI for depth.

3\. Use single-shot importance estimation; iterative provides no benefit.

4\. Prefer width pruning over depth for the model scales we consider (≤ 15B).

5\. Retrain exclusively with distillation loss using KLD instead of conventional training.

6\. Use (logit+intermediate state+embedding) distillation when depth is reduced significantly.

7\. Use logit-only distillation when depth isn’t reduced significantly.

8\. Prune a model closest to the target size.

9\. Perform lightweight retraining to stabilize the rankings of searched pruned candidates.

10\. If the largest model is trained using a multi-phase training strategy, it is best to prune and

retrain the model obtained from the final stage of training

### Best Practices



1. Iterative Importance nie daje benefitu
2. Width-only Pruning jest najlepszy
3. Distillation lepszy niż training from scratch (i w nim KL-divergence jako loss)
4. Iterative Pruning i Distillation across model sizes daje benefity








