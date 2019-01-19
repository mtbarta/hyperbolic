# hyperbolic
NLP models in hyperbolic space

this is a collection of work i've done to investigate the use of hyperbolic space for NLP tasks. I'm building on top of @dpressel 's baseline project.

I'm using word embeddings generated from [Leimeister (2018)](https://arxiv.org/pdf/1809.01498.pdf)

### Overall Learnings
---
- learning the softmax translation as shown by Ganea 2018 is required.
- double precision is needed to successfully learn. 
- I found relu activations to be helpful in sequential tagging.
- A manifold abstraction would be helpful, as the code gets quite messy and doesn't lend itself to changing manifolds easily.

### Tagging
---
- [x] basic tagger in the Poincare Ball, leveraging [Ganea 2018](https://arxiv.org/pdf/1805.09112.pdf)

- the hyperboloid RNN has a vanishing gradient and needs more exploration.
- bidirectional models in the hyperboloid need more investigation. I thought I would have seen a larger gain from this similar to euclidean space.

### Classification
---
I used Ganea's work to apply it to the AG news dataset, but was never certain it worked correctly.

references
---
https://arxiv.org/pdf/1805.09112.pdf  
https://arxiv.org/pdf/1805.08207.pdf  
https://arxiv.org/pdf/1806.03417.pdf  

https://github.com/lateral/minkowski/blob/master/python/hyperboloid_helpers/manifold.py  
https://github.com/geomstats/geomstats/blob/master/geomstats/riemannian_metric.py  
https://github.com/dpressel/baseline  