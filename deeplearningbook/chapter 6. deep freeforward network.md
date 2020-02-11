

## 6. 심층 순방향 신경망



- <u>심층 순방향 신경망</u> (deep freeforward network): 심층학습 모형의 정수

- <u>다층 퍼셉트론</u> (MLP) : 순방향 신경망의 목표 

- 모형의 앞쪽으로만 흘러가고 피드백이 없기 때문에 순방향이라고 한다. 

- 특수한 버전은 CNN, 자연어처리의 바탕기술

  

> For example, we might have three functions *f(1), f(2), and f(3)* connected in a chain, to form *f(x) =f(3)(f(2)(f(1)(x)))*. These chain structures are the most commonly used structures of **neural networks**.

------



- f^(1)^ 을 신경망 제 1층, f^(2)^ 를 제 2층이라 하며 사슬의 전체길이는 **모형의 깊이**에 해당한다.
- 순방향 신경망의 마지막 층을 **출력층(output layer)** 라고 부른다.
- 가장 좋은 근사값이 산출되도록 하는 층들의 출력을 훈련자료가 보여주지 않는다는 점에서 그런 층들을 **은닉층(hidden layer)**라고 부른다.
- 은닉층은 벡터를 입력받는데 그때 벡터의 차원은 신경망의 너비를 결정한다.

------



> 순방향 신경망을 이해하는 방법으로 선형모형의 한계를 어떻게 극복할지 고민해보는것이며 선형모형을 비선형 모형으로 확장하는 한 방법은 커널 트릭을 사용하는 것.



------



> 순방향 신경망을 설계하려면

1. hidden layer의 값을 계산하는데 사용할 활성화 함수가 필요.
2. 신경망 구조 설계



## 6.1 Example: Learning XOR

XOR 함수는 두 이진수 x~1~, x~2~에 대한 연산이다. 두 이진수 중 정확히 하나가 1과 같으면 XOR 함수는 1을 돌려주고, 그렇지 않으면 0을 돌려준다.

<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200205234757350.png" alt="image-20200205234757350" style="zoom:67%;" />



 XOR 문제는 매우 단순하지만 선형적으로 구분할 수 없다. 그래서 이를 해결하기 위해 인간의 뉴런을 모방한 퍼셉트론을 층층이 쌓는 구조(Multilayer perceptrons, multilayer neural nets)가 제시 됐다.

이 신경망 층을 학습시킬 방법으로 **역전파(Backpropagation)** 알고리즘이 제시됐으며, 또 한가지 입력을 엄청나게 많은 단계로 나눠 분석하는 Convolutional Neural Networks(합성곱 신경망)모델 또한 제시됐다.

하지만 역전파 알고리즘의 효과가 현실의 문제를 다루기 위해 계층의 수를 늘려갈수록 희미해지는 문제가 생겼다.
그와 동시에 **SVM(Support Vector Machine), 의사결정나무와 Random Forest** 등의 학습 알고리즘들이 신경망의 대체제로 대두됐다.

<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200205235330012.png" alt="image-20200205235330012" style="zoom:67%;" />

- x~1~=0 일 때는 x~2~ 가 증가함에 따라 모형의 출력도 증가해야 한다.
- x~1~=1 일 때는 x~2~ 가 증가함에 따라 모형의 출력이 감소해야 한다.
- 위 경우 둘 다 포함하는 한 점 h=[1,0]^T^ , h~1~이 증가하면 h~2~가 감소하는 함수  



<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206003059945.png" alt="image-20200206003059945" style="zoom:67%;" />



- XOR 학습예제의 순방향 신경망을 두 가지 스타일로 표현한 모습. 이 신경망에는 은닉단위가 두 개인 은닉층이 있다.

- h 는  f^(1)^ $\ (x; W, c)$ 로 계산 되는 hidden units 들의 벡터

- h = f ^(1)^ $\ (x; W , c)$ and y= f ^(2)^ $\ (h; w, b)$, with the complete model being 
  $\ f(x; W , c, w, b)$ = f ^(2)^  ( f ^(1)^(x))

- f ^(1)^이 선형이면 순방향 신경망 전체가 주어진 입력의 선형함수가 된다.

- 따라서 비선형 함수를 이용해야 하고 활성화 함수는  주로 **ReLU**가 쓰인다.

  

<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206010513365.png" alt="image-20200206010513365" style="zoom:67%;" />

- 이 함수를 선형변환의 출력에 적용하면 하나의 비선형 변환이 나온다.

- ReLU는 선형 함수에 가까우며 기울기 기반 방법들로 최적화하기 쉽게 만드는 여러 성질과 일반화가 잘 되게 만드는  선형함수의 여러 성질이 남아 있다.

  

  <img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206141445284.png" alt="image-20200206141445284" style="zoom:67%;" />



<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206142253323.png" alt="image-20200206142253323" style="zoom:67%;" />



![image-20200206141149001](C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206141149001.png)

- 선형 모형으로는 0에서 1로 올랐다가 1에서 다시 0으로 떨어지는 함수를 구현할 수 없기 때문에 정류선형 변환(rectified linear transformation)을 적용
- 실제 상황에서는 데이터가 매우 크기 때문에 이와 같이 해를 바로 도출하는건 힘들 수도 있음.
  
  

## 6.2 기울기 기반 학습

- 신경망은 비선형성이기 때문에 볼록함수를 손실함수로 사용하는게 적합하지 않을 때가 많다.
- 일반적으로 신경망에서는 볼록함수 최적화 알고리즘 대신 비용함수를 아주 낮은 값으로 이끄는 역할만 하는 반복적인 기울기 기반 최적화 절차를 사용한다.
- 순방향 신경망에서는 모든 가중치를 작은 난수들로 초기화 하는 것이 중요



## 6.2.1 비용함수

- 대부분의 경우 하나의 분포 $\ p(y | x; θ)$를  정의하고 최대가능도 원리를 적용해서 훈련을 진행한다. 이런 경우 훈련자료와 모형의 예측 사이의 크로스 엔트로피를 비용함수로 사용하면 된다.

- 신경망 훈련에 쓰이는 총비용함수(total cost function)는 여기서 설명하는 기본적인 비용함수에 정칙화 항을 결합한 형태 일 때가 많다.
  
  

  ### 6.2.1.1 최대가능도를 이용한 조건부 확률 학습

  <img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206154329835.png" alt="image-20200206154329835" style="zoom:67%;" />

  > 대부분의 현대적 신경망은 최대가능도를 사용해서 훈련한다. 이는 비용함수가 음의 로그가능도라는 뜻이다.

  - 최대가능도의 한 가지 장점은 모형마다 매번 비용함수를 설계하는데 부담이 없다는 것이다. 모형 $\ p(y|x)$만 결정되면 비용함수 $\ logp(y|x)$가 자동으로 결정된다.
  - 비용함수의 기울기는 학습 알고리즘을 잘 지도 할 수 있을 정도로 크고 예측 가능해야 한다. 
  - 활성화 함수가 포화하면 비용함수의 기울기가 아주 작아진다. 그런 현상을 피하는 데 음의 로그 가능도가 도움이 되는 모형이 많다.
  - 교차 엔트로피 비용함수의 한 가지 독특한 성질은, 실제 응용에 흔히 쓰이는 모형들에서 최솟값이 없을 때가 많다는 것이다.



### 	6.2.1.2 조건부 통계량의 학습 

​	 전체 확률분포 $\ p(y|x;θ) $를 배우는 것이 아니라 $\ x$가 주어졌을 때$\ y$의 한 조건      	부 통계량만 배우면 되는 경우도 있다.
​	 
​	 비용함수를 범함수로 간주 할 수 있는데

![image-20200206164028471](C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206164028471.png)

​	특정 형태의 함수의 파라미터 구성을 선택하는 것이 아니라 함수 자체를 선택 	하는 것이라고 보면 된다,,,



<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206164431398.png" alt="image-20200206164431398" style="zoom:67%;" />

​	  이러한 비용함수를 흔히 평균절대오차(mean absolute error)라고 부른다. 안	  타깝게도 평균제곱오차나 평균 절대오차를 기울기 기반 최적화와 함께 사용	  하면 성능이 나쁠때가 많다. 이것이 크로스 엔트로피를 비용함수로 많이 쓰는 	  이유 중 하나이다.



## 6.2.2 출력 단위

- 비용함수의 선택은 출력 단위의 선택과 밀접하게 관련되어 있다.

- 대부분의 경우에는 자료 분포와 모형 분포 사이의 크로스 엔트로피를 비용함수로 사용한다.

- 출력층(output layer)의 역할은 적절한 변환으로 신경망이 풀어야 할 과제를 만족하는 출력을 산출하는 것이다. 

  

  ### 6.2.2.1 가우스 출력 분포를 위한 선형 단위

  > 선형 출력 단위들로 이루어진 선형 출력층은 주어진 특징 h로 부터 벡터       <img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206165905289.png" alt="image-20200206165905289" style="zoom: 50%;" /> 를 산출하고 이런 종류의 출력 단위를 선형단위(linear unit)이라고 부를때가 많다.

  <img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206170947934.png" alt="image-20200206170947934" style="zoom:67%;" />

- 이 경우 로그가능도를 최대화 하는 것은 평균제곱오차를 최소화 하는 것과 같다.

- 가우스 분포의 공분산도 간단하게 학습할 수 있고, input function으로 만드는 것도 가능하다.

- 공분산 행렬이 항상 양의 정부호 행렬이여야 한다는 제약 때문에 선형 이외의 것을 사용하는 것이 좋지만 선형단위들은 포화하지 않으므로 기울기 기반 최적화 알고리즘에서 문제를 별로 일으키지 않으며, 그 밖에도 다양한 최적화 알고리즘들과 잘 작동한다. 

  

  ### 6.2.2.2 베르누이 출력분포를 위한 sigmoid units

  > 심층학습 과제 중에는 이진 변수 y의 값을 예측하는 것이 많다. 이러한 종류 문제들의 최대가능도 접근 방식을 적용할 때는 x가 주어졌을 때의 y에 관한 베르누이 분포를 정의한다.

-  신경망은 $\ p(y=1|x)$ 만 예측하면 된다. 확률이기 때문에 [0,1]이다.

- output layer가 선형 단위 하나로 이루어진다고 한다. 그 단위 값이 [0,1]에 속하려면 다음과 같이 범위를 한정할 필요가 있다.

  <img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206205608616.png" alt="image-20200206205608616" style="zoom:67%;" />

- W^T^h+b  가 단위구간을 벗어날 때마다 모형 출력의 기울기는 0이 된다. 기울기가 0이면 학습 알고리즘이 파라미터가 개선되는 방향을 결정할 수 없기 때문에 훈련에 문제가 생긴다.

- 모형이 잘못된 답을 낼 때마다 항상 강한 기울기를 산출하는 다른 접근 방식이 필요하다.

  <img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206215433019.png" alt="image-20200206215433019" style="zoom:67%;" />

  여기서 $\sigma$는 로그 시그모이드 함수이다.

- z= W^T^h+ b 를 계산한뒤 시그모이드 활성화 함수를 통해 z를 하나의 확률값으로 변환한다.

- 시그모이드 함수는 정규화가 되지 않은, 즉 확률의 합이 1이 아닌 확률분포를 구축할 때 필요하다.

- 정규화 되지 않은 y와 z에서 선형이라고 가정할 때, 그 로그 확률들을 거듭제곱 하면 정규화 되지 않은 확률들이 나온다. 그 확률들을 적당한 상수로 나누어서 정규화 하면 유효한 확률분포가 나온다.

<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206221900928.png" alt="image-20200206221900928" style="zoom:67%;" />

- z변수를 logit이라고 부른다.

<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206222526888.png" alt="image-20200206222526888" style="zoom:67%;" />



<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200206223013712.png" alt="image-20200206223013712" style="zoom:67%;" />

<img src="C:\Users\hyojung\AppData\Roaming\Typora\typora-user-images\image-20200207003919764.png" alt="image-20200207003919764" style="zoom:67%;" />

- z가 극도로 부정확한 경우가 아니라면 소프트플러스 함수의 기울기가 아주 작아지는 일은 없다. 기울기 기반 학습 알고리즘이 잘못된 z 값을 즉시 바로잡는 행동을 보일 것이라는 점에서 이는 유용한 성질이다.
- 포화가 발생하면 학습에서 유용하지 않을 정도로 기울기가 작아질 수 있다.
- 시그모이드 활성화 함수는 z가 매우 작을때는 0으로 포화하고 z가 아주 큰 양수일 때는 1로 포화한다. 따라서, 시그모이드 출력 단위들이 있는 신경망에서는 최대가능도가 바람직한 접근 방식이다.
