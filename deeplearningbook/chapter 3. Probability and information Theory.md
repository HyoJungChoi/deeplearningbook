

### why probability?

- machine learning makes heavy use of probability theory.
- machine learning always deal with uncertain quantities and sometimes stochastic quantities.



### Multivariate normal distribution

![3-1](https://user-images.githubusercontent.com/56706812/72909238-06f14100-3d7a-11ea-861a-d8ac94206e09.png)

- covariance matrix $\Sigma$ : p.d symmetric matrix = all positive eigenvalues
- precision matirix $\beta$  = $\Sigma$ ^-1^
- we often fix the covariance matrix to be a diagonal matrix.
- isotropic Gaussian distribution : scalar * I~n~



### Laplace Distribution

![3-2](https://user-images.githubusercontent.com/56706812/72909293-20928880-3d7a-11ea-8da5-0009cdfcdc9f.png)

![3-3](https://user-images.githubusercontent.com/56706812/72910584-140f2f80-3d7c-11ea-9f74-1bbd943d5b12.png)



### The Dirac Distribution 

![3-4](https://user-images.githubusercontent.com/56706812/72910630-26896900-3d7c-11ea-9256-6037cacd412a.png)

- when we wish to specify that all the mass in a probability distribution clusters around a single point. This can be accomplished by defining a PDF using the Dirac delta function
- it is zero valued everywhere except 0, yet integrates to 1.

![3-5](https://user-images.githubusercontent.com/56706812/72910700-3f921a00-3d7c-11ea-85cb-14107287124d.png)

- generalized function is defined in terms of it properties when integrated.



### Empirical Distribution

![3-6](https://user-images.githubusercontent.com/56706812/72910766-5afd2500-3d7c-11ea-9b80-6f83783609b6.png)

- we obtain an inﬁnitely narrow and inﬁnitely high peak of probability density where x = µ.
- The Dirac delta distribution is only necessary to deﬁne the empirical distribution over continuous variables.
- For discrete variables, the situation is simpler: an empirical distribution can be conceptualized as a multinoulli distribution.
- important perspective on the empirical distribution is that it is the probability density that maximizes the likelihood of the training data



### Mixtures of distribution

- The mixture model is one simple strategy for combining probability distributions to create a richer distribution.



### Latent variables

- random variable that we cannot observe directly.
-  P(x | c) relating the latent variables to the visible variables determines the shape of the distribution P(x), even though it is possible to describe P(x) without reference to the latent variable.



### Gaussian mixture model

- A very powerful and common type of mixture model .

- As with a single Gaussian distribution, the mixture of Gaussians might constrain the covariance matrix for each component to be diagonal or isotropic.

- universal approximator of densities.

  

  ![3-6](https://user-images.githubusercontent.com/56706812/72910766-5afd2500-3d7c-11ea-9b80-6f83783609b6.png)

  

### logistic sigmoid

![3-8](https://user-images.githubusercontent.com/56706812/72910932-97c91c00-3d7c-11ea-880f-e9276e3ffbb2.png)

- The logistic sigmoid is commonly used to produce the φ parameter of a Bernoulli distribution because its range is (0,1).
- sigmoid function saturates when its argument is very positive or very negative, meaning that the function becomes very ﬂat and insensitive to small changes in its input.





### Softplus function

- useful for producing the β or σ parameter of a normal distribution because its range is (0, ∞).

- softened or smoothed version of

![3-9](https://user-images.githubusercontent.com/56706812/72910993-ae6f7300-3d7c-11ea-8a22-7d61d153a283.png)

![3-10](https://user-images.githubusercontent.com/56706812/72911021-bb8c6200-3d7c-11ea-8ef7-86d809b6503f.png)



### Measure Theory

- A proper formal understanding of continuous random variables and probability density functions requires developing probability theory in terms of a branch of mathematics known as measure theory.
- One of the key contributions of measure theory is to provide a characterization of the set of sets we can compute the probability of without encountering paradoxes.
- provides a rigorous way of describing that a set of points is negligibly small. Such a set is said to have measure zero.
- Another useful term from measure theory is almost everywhere (a.e.)



### Information theory

1. calculate the expected length of messages.
2.  characterize probability distributions .
3. to quantify similarity between probability distributions.



### Self-information

![3-11](https://user-images.githubusercontent.com/56706812/72911054-c941e780-3d7c-11ea-89da-6592a4261cd1.png)

- Our deﬁnition of I (x) is therefore written in units of nats.

-  One nat is the amount of information gained by observing an event of probability 1/e.

- use base-2 logarithms and units called bits or shannons.

- Self-information deals only with a single outcome.

  

### Shannon entropy

![3-12](https://user-images.githubusercontent.com/56706812/72911112-dfe83e80-3d7c-11ea-8d91-fdc0ffe72dfe.png)

- We can quantify the amount of uncertainty in an entire probability distribution using the Shannon entropy.

- Distributions that are nearly deterministic (where the outcome is nearly certain) have low entropy; distributions that are closer to uniform have high entropy.

- When x is continuous,Shannon entropy is known as the diﬀerential entropy.

  ![3-13](https://user-images.githubusercontent.com/56706812/72911143-ec6c9700-3d7c-11ea-8b45-dc8e2ec60071.png)

- When p is near 0, the distribution is nearly deterministic When p is near 1, the distribution is nearly deterministic, because the random variable is nearly always 1. When p= 0.5, the entropy is maximal, because the distribution is uniform over the two outcomes.

  

### Kullback-Leibler (KL) divergence

![3-14](https://user-images.githubusercontent.com/56706812/72911180-f7bfc280-3d7c-11ea-9e0a-8e3141da7504.png)

- most notably being non-negative.

- 0 if and only if P and Q are the same distribution in the case of discrete variables, or equal “almost everywhere” in the case of continuous variables.

- It is not a true distance measure .

  ![3 15](https://user-images.githubusercontent.com/56706812/72911214-04dcb180-3d7d-11ea-8638-e20c92f261fb.png)



### Cross entropy

##### H(P, Q)= H(P) + D~KL~(P||Q)   =>  KLD= cross entropy- shannon entropy

- Minimizing the cross-entropy with respect to Q is equivalent to minimizing the

  KL divergence, because Q does not participate in the omitted term.



### Structured Probablistic Model

![3-16](https://user-images.githubusercontent.com/56706812/72911308-28076100-3d7d-11ea-951f-b65a745e80db.png)

- These factorizations can greatly reduce the number of parameters needed

  to describe the distribution.

- When we represent the factorization of a probability distribution with a graph, we call it a structured **probabilistic model, or graphical model.**

- **Directed**
  
  ![3-17](https://user-images.githubusercontent.com/56706812/72911327-32295f80-3d7d-11ea-9917-dd381a8f5c85.png)

![3-18](https://user-images.githubusercontent.com/56706812/72911397-57b66900-3d7d-11ea-9c50-9855cf41a455.png)

- they represent factorizations into conditional probability distributions.
-  a directed model contains one factor for every random variable x i in the distribution.
- **Undirected**

![3-19](https://user-images.githubusercontent.com/56706812/72911417-61d86780-3d7d-11ea-9984-f00d6932e35c.png)

![3-20](https://user-images.githubusercontent.com/56706812/72911437-6d2b9300-3d7d-11ea-8c5a-24816796411a.png)

- graphs with undirected edges.
- these functions are usually not probability distributions of any kind.

