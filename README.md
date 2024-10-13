# What is this?

Suppose a container contains rocks of different type, which are loaded in batches. There are at most three slices, each having a type (a number from 1 to 4), and a weight.
![[Pasted image 20241013233639.png]]

We want to compute the weighted average type of the container:
$$
average = w1*t1 + w2*t2 + w3*t3.
$$
The problem is that we do not know the actual types $t1,t2,t3$: we measure them, but the measurements maybe imprecise. Therefore, given a table of examples of weights-types and resulting averages, we want to learn the function `real_average`.

- `generate_data.py` generates a tab-separated data in the format
```
t1  t2  t3  w1  w2  w3  diff  average  perfect_average  real_average
```
  where  `t1,t2,t3` are numbers from 1 to 4, `w1,w2,w3` are numbers from 0 to 1, `diff` is the difference `real_average - average`, `average` is the weighted average, and `perfect_average` is the real average when we remove Gaussian-distributed error. _Note: the learning algorithm does not have access to the last two columns, they are for convinience._
- `all_graphs.py` generates many useful graphs for visual inspection of the data.
- `heat_map.py` generates heat maps for the visual inspection of the data.
- `regression_with_hyperopt.py` searches for the best regression model using `hyperopt` and `sklearn` and produces data.

# Running example

1. `python3 ./generate_data.tsv > data.tsv`
2. `python3 ./all_graphs.py ./data.tsv`

![[all_graphs.png]]

3. `python3 ./heat_map.py ./data.tsv`
![[heat_map.png]]
4. `python3 ./regression_with_hyperopt.py ./data.tsv`

![[results.png]]
After about 5 minutes, a decent regression model is found: 
```
{'learner': SVR(C=1.248634607901186, coef0=0.7928013919304228, degree=2,
    epsilon=0.05528341011338442, gamma='auto', kernel='poly',
    tol=1.890764067171372e-05), 'preprocs': (), 'ex_preprocs': ()}
    
MSE train: 0.01
MSE test: 0.02

MAE train: 0.09
MAE test: 0.10

R^2 train: 0.848, R^2 test: 0.823
```
The very last plot (bottom-right) shows the results of the learned-model predictions on _all_ data. The plots on the left compare it with the naive weighted average model and the perfect model (which knows the exact -- not measured -- values of types, but which of course cannot predict the Gaussian noise). As you can see, the learned model is close to the perfect possible one. In the first row of plots, there are two plots: learned differences (`model_predict_average - average`) vs actual differences (`actual_average - average`). (There are only two plots in the first row.) The gray lines plot the functions `y=0.9x, y=x, y=1.1x`.

