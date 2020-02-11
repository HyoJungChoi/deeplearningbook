## chapter 5. Machine Learning Basics

-  <u>Deep learning</u> is a speciﬁc kind of machine learning.
- Most machine learning algorithms can be divided into the categories of supervised learning and unsupervised learning.



## 5.1 Leaning Algorithms

### 5.1.1 The task T

- Machine learning tasks are usually described in terms of how the machine learning system should process an example.

- We typically represent an example as a vector x ∈ R^n^ where each entry

  xi of the vector is another feature.
  
  

#### <u>Classification</u> : 

In this type of task, the computer program is asked to **specify which of k categories some input belongs to.** To solve this task, the learning algorithm is usually asked to produce a function f: Rn→ {1, . . . , k}.   for example, where f outputs a probability distribution over classes. An example of a classiﬁcation task is **object recognition**, where the input is an image.



#### <u>Classiﬁcation with missing inputs</u>:

When some of the inputs may be missing, rather than providing a **single classiﬁcation function, **the learning algorithm must learn a set of functions. Each function corresponds to classifying x with a diﬀerent subset of its inputs missing. One way to eﬃciently deﬁne such a **large set** of functions is to learn a probability distribution over all the relevant variables, then solve the **classiﬁcation task by marginalizing out the missing variables.** but the computer program needs to learn only a single function describing the joint probability distribution. *ex) medical diagnosis*



#### <u>Regression:</u>

To solve this task, the learning algorithm is asked to output a function f: R^n^→ R. **This type of task is similar to classiﬁcation, except that the format of output is diﬀerent.** These kinds of predictions are also used for algorithmic trading. *ex) prediction of expected claim amount, prediction of future prices of securities.*



#### <u>Transcription:</u>

In this type of task, the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe the information into discrete textual form. *ex) character recognition, speech recognition*



#### <u>Machine translation:</u> 

In a machine translation task, **the input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language.** This is commonly applied to natural languages.



#### <u>Structured output:</u>

Structured output tasks involve any task where the output is a vector (or other data structure containing multiple values) with important relationships between the diﬀerent elements. This is a broad category and subsumes the transcription and translation tasks described above, as well as many other tasks. *ex) parsing—mapping a natural language sentence into a tree, pixel-wise segmentation of images*



#### <u>Anomaly detection:</u>

In this type of task, the computer program sifts through a set of events or objects and ﬂags some of them as being unusual or atypical. *ex) credit card fraud detection*



#### <u>Synthesis and Sampling:</u>

In this type of task, the machine learning algorithm is asked to generate new examples that are similar to those in the training data. Synthesis and sampling via machine learning can be useful for media applications when generating large volumes of content by hand would be expensive, boring, or require too much time. **This is a kind of structured output task, but with the added qualiﬁcation that there is no single correct output for each input,** and we explicitly desire a large amount of variation in the output, **in order for the output to seem more natural and realistic**   *ex) video games, speech synthesis task*



#### <u>Imputation of missing values</u>:

In this type of task, the machine learning algorithm is given a new example x ∈ R^n^ , but with some entries xi of x missing. **The algorithm must provide a prediction of the values of the missing entries.**



#### <u>Denoising:</u>

In this type of task, the machine learning algorithm is given as input a corrupted example˜x ∈ R^n^ obtained by an unknown corruption process from a clean example x ∈ R^n^ . **The learner must predict the clean example x from its corrupted version˜x , or more generally predict the conditional probability distribution p(x |˜x).**



#### <u>Density estimation or probability mass function estimation</u>:

Most of the tasks described above require the learning algorithm to at least implicitly capture the structure of the probability distribution. **Density estimation enables us to explicitly capture that distribution.** In principle, we can then perform computations on that distribution to solve the other tasks as well. 



### 5.1.2 The Performance Measure, P



- For tasks such as classiﬁcation, classiﬁcation with missing inputs, and transcription, we often measure the **accuracy** of the model.
- We can also obtain equivalent information by measuring the **error rate**, the proportion of examples for which the model produces an incorrect output.
- We therefore evaluate these performance measures **using a test set** of data that is separate from the data used for training the machine learning system.



### 5.1.3 The Experience, E



#### <u>Unsupervised learning algorithms</u>:

experience a dataset containing many features, **then learn useful properties of the structure of this dataset.** In the context of deep learning, we usually want to learn the entire probability distribution that generated a dataset, whether explicitly, as in density estimation, or implicitly, for tasks like synthesis or denoising. *ex) clustering*

#### <u>Supervised learning algorithms</u>:

unsupervised learning involves observing several examples of a random vector x and attempting to implicitly or explicitly learn the probability distribution p(x), or some interesting properties of that distribution;

Many machine learning technologies can be used to perform both tasks. For example,

<img src="https://user-images.githubusercontent.com/56706812/74208880-bf027180-4cc8-11ea-8259-28fe4a0202d9.png" alt="image-20200130162048616" style="zoom:67%;" />

Alternatively, we can solve the supervised learning problem of learning     p(y | x) by using traditional unsupervised learning technologies to learn the joint distribution p (x, y), then inferring

<img src="https://user-images.githubusercontent.com/56706812/74208900-d04b7e00-4cc8-11ea-9592-fd631fbf0dbc.png" alt="image-20200130162027289" style="zoom:67%;" />





#### <u>reinforcement learning algorithms</u>:

do not just experience a ﬁxed dataset. there is a feedback loop between the learning system and its experiences.



#### <u>design matrix</u>:

$$
Y=X\beta + \epsilon
$$

A design matrix is a matrix containing a diﬀerent example in each row. Each column of the matrix corresponds to a diﬀerent feature. it must be possible to describe each example as a vector, and each of these vectors must be the **same size**.



### 5.1.4 Example : Linear Regression

 the goal is to **build a system that can take a vector x ∈ R^n^ as input and predict the value of a scalar y ∈ R as its output.** Let y hat be the value that our model predicts y should take on. We deﬁne the output to be



<img src="https://user-images.githubusercontent.com/56706812/74209066-8fa03480-4cc9-11ea-9775-840000e33c53.png" alt="image-20200128174318531" style="zoom:67%;" />



One way of measuring the performance of the model is to compute the **mean squared error** of the model on the test set.

<img src="https://user-images.githubusercontent.com/56706812/74209073-9a5ac980-4cc9-11ea-834d-ddae417e634f.png" alt="image-20200128174623693" style="zoom:67%;" />

we can also see that

<img src="https://user-images.githubusercontent.com/56706812/74209102-a9417c00-4cc9-11ea-8fcf-4370f45b32dc.png" alt="image-20200128174721348" style="zoom:67%;" />

We need to design an algorithm that will improve the weights w in a way that **reduces MSE test** when the algorithm is allowed to gain experience by observing a training set (X (train), y(train)). To **minimize MSE train, we can simply solve for where its gradient is 0**.



<img src="https://user-images.githubusercontent.com/56706812/74209139-d8f08400-4cc9-11ea-8a33-917fbc5f372d.png" alt="image-20200128175008940" style="zoom:67%;" />

<img src="https://user-images.githubusercontent.com/56706812/74209154-e574dc80-4cc9-11ea-8960-60dd795b558b.png" alt="image-20200128175359066" style="zoom:67%;" />



**5.12** is known as the **normal equation**.



### 5.2 Capacity, Overﬁtting and Underﬁtting



- The ability to perform well on previously unobserved inputs is called **generalization**.

- we can compute some error measure on the training set, called the **training error**; and we reduce this training error.

- What separates machine learning from optimization is that we want the **generalization error**, also called the **test error**, to be low as well.

In our linear regression example, we trained the model by minimizing the

training error, 

<img src="https://user-images.githubusercontent.com/56706812/74209169-f6255280-4cc9-11ea-8f29-e48636d56c0c.png" alt="image-20200128205934457" style="zoom:67%;" />



using statistical learning theory, The training and test data are generated by a probability distribution over datasets called the **data-generating process**   

1.  make a set of assumptions known collectively as the **i.i.d. assumptions**. These assumptions are that the examples in each dataset are **independent** from each other

2. the training set and test set are  **identically distributed**, drawn from the same probability distribution as each other.

3. We call that shared underlying distribution the **data-generating distribution**, denoted p~data~.
4. For some ﬁxed value **w**, the expected training set error is exactly the same as the expected test set error, because both expectations are formed using the same dataset sampling process.



The factors determining how well a machine learning algorithm will perform are its **ability** to

1. *Make the training error small.*
2. *Make the gap between training and test error small.*



***Underﬁtting*** occurs when the model is not able to obtain a suﬃciently low error value on the training set. 

***Overﬁtting*** occurs when the gap between the training error and test error is too large.

We can control whether a model is more likely to overﬁt or underﬁt by altering its capacity. Models with **low capacity may struggle to ﬁt the training set**. Models with **high capacity can overﬁt by memorizing properties of the training set** that do not serve them well on the test set.

Models with **insuﬃcient capacity are unable to solve complex tasks**. Models with **high capacity can solve complex tasks,** but when their capacity is higher than needed to solve the present task, they may **overﬁt**.



<img src="https://user-images.githubusercontent.com/56706812/74209186-063d3200-4cca-11ea-8069-5bebeb7a6c7a.png" alt="image-20200128232115012" style="zoom:67%;" />



**(Left)** A linear function ﬁt to the data suﬀers from underﬁtting—it cannot capture the curvature that is present in the data. 

**(Center)** A quadratic function ﬁt to the data generalizes well to unseen points. It does not suﬀer from a signiﬁcant amount of overﬁtting or underﬁtting. 

**(Right)** A polynomial of degree 9 ﬁt to the data suﬀers from overﬁtting.



<img src="https://user-images.githubusercontent.com/56706812/74209255-5916e980-4cca-11ea-8cda-82b5b1117558.png" alt="image-20200128235045126" style="zoom:67%;" />



At the **left end of the graph**, training error and generalization error are both high. This is the **underﬁtting regime.**

Eventually, the size of this gap outweighs the decrease in training error, and we enter the **overﬁtting regime**, where capacity is too large, above the **optimal capacity** .

The error incurred by an oracle making predictions from the true distribution p(x, y) is called the **Bayes error**.

it is possible for the model to have optimal capacity and yet still have a **large gap between training and generalization errors.** In this situation, we may be **able to reduce this gap by gathering more training examples**.



<img src="https://user-images.githubusercontent.com/56706812/74209327-9ed3b200-4cca-11ea-8924-c59881fd63f8.png" alt="image-20200129002249420" style="zoom:67%;" />

**Top)** The MSE on the training and test set for two diﬀerent models: a quadratic model, and a model with degree chosen to minimize the test error. Both are ﬁt in closed form. For the quadratic model, **the training error increases as the size of the training set increases.** This is because larger datasets are harder to ﬁt. Simultaneously, the test error decreases, because fewer incorrect hypotheses are consistent with the training data.

**(Bottom)** As the **training set size increases, the optimal capacity increases**. The optimal capacity plateaus after reaching suﬃcient complexity to solve the task



### 5.2.1 The No Free Lunch Theorem

 **The no free lunch theorem** for machine learning (Wolpert, 1996) states that, averaged over all possible data-generating distributions, every classiﬁcation algorithm has the same error rate when classifying previously unobserved points.

our goal is to understand what kinds of distributions are relevant to the “real world” that an AI agent experiences, and what kinds of machine learning algorithms perform well on data drawn from the kinds of data-generating distributions we care about.





### 5.2.2 Regularization

To perform linear regression with **weight decay**, we minimize a sum **J (w)** comprising both the mean squared error on the training and a criterion that expresses a preference for the weights to have smaller squared L^2^ norm.

<img src="https://user-images.githubusercontent.com/56706812/74209343-ac893780-4cca-11ea-85cc-ecd7fd111fcf.png" alt="image-20200129143704585" style="zoom:67%;" />

**Minimizing J(w)** results in a choice of weights that make a tradeoﬀ between ﬁtting the training data and being small. This gives us solutions that have a smaller slope, or that put weight on fewer of the features

<img src="https://user-images.githubusercontent.com/56706812/74209365-c3c82500-4cca-11ea-9420-cee837815b32.png" alt="image-20200129144118134" style="zoom:67%;" />

More generally, we can regularize a model that learns a function f(x;θ) by **adding a penalty** called a regularizer to the cost function. In the case of weight decay, the regularizer is Ω(w) = w^T^w.

Regularization is any modiﬁcation we make to a learning algorithm that is intended to **reduce its generalization error but not its training error. **Regularization is one of the central concerns of the ﬁeld of machine learning, rivaled in its importance only by optimization