# Variational Autoencoder re-implementation

A reproduction of the variational autoencoder described in (Kingma and Welling, 2013). 

## 3 line summary
Variational autoencoders is a neural architecture that takes a probability distribution (aka data), 
and tries to find a transformation on that distribution such that the transformed (latent) distribution follows a normal distribution.
Simultaneously, it also learns a transformation from the generated normal distribution back into the original distribution.
Both the transformations to the latent distribution and back to the data distribution are both done through neural networks,
and as such, the entire model can be trained end to end using backpropagation, as the entire model is differentiable
(provided that the reparameterization trick described in the paper is used).

This was mostly done as a learning experiment to gain a deeper understanding of the methods used in the paper through
understanding the exact comptations used and being able to play with and understand the outputs generated.

https://arxiv.org/pdf/1312.6114.pdf

Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
APA	
