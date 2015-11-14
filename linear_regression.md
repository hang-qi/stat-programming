# Linear Regression

Suppose we have a dataset consisting of $$n$$ observations, each of which has $$p$$ predictors / feature.
As training data, each of the $$n$$ examples has an outcome $$y_i$$. In the classic scenarios, $$p$$ is usually a fixed number and $$ n \gg p $$. Whereas in mordern cases, we usually have more features $$p > n$$.

| obs |$$X_{n\times p}$$| $$Y_{n\times 1}$$ |
|-----|-----------|-----|
| 1   |    $$x_1^T$$  |$$y_1$$|
| 2   |    $$x_2^T$$  |$$y_2$$|
| ... |           |     |
| n   |    $$x_n^T$$  |$$y_n$$|

In the table, each row $$x_i^T = ( x_{i1}, x_{i2} , \ldots, x_{ip})$$ is a data point and its corresponding outcome $$y_i$$.
If we assume linear relationships between the predictors and outcome with coefficients $$\beta = (\beta_1, \beta_2\, \ldots, \beta_p)^T$$, we have linear regression
$$
y_i = \beta_1 x_{i1} + \beta_2 x_{i2}  + \cdots +  \beta_p x_{ip} + \epsilon_i = x_i^T \beta + \epsilon_i.
$$

To find $$\beta$$, we minimize the residue sum of squares 
$$
R(\beta) = \sum_{i=1}^{n} \left(y_i - x_i^T\beta\right)^2.
$$

$$
\hat{\beta} = \arg\min_{\beta}R(\beta).
$$

In the matrix form, suppose the data matrix $$X_{n\times p} = (X_1, X_2, \ldots, X_p)$$ is composed by $$p$$ columns in the table above, $$Y$$ is the vector which collects all the outcomes, we can write the residue as
$$
R(\beta) = \left| Y - X\beta \right|^2.
$$

This means, we are looking for an approximated $$\hat{Y} = X\hat{\beta}$$ in the space spanned by column vectors of $$X$$, i.e. $$X_1, X_2, \ldots, X_p$$, so that the residue $$R(\hat{\beta}) = Y^TY - \hat{Y}^T \hat{Y}$$, which is the squared distance between $$\hat{Y}$$ and $$Y$$, is minimized. We can solve a close form solution

$$
\hat{\beta} = \left(X^T X\right)^{-1}X^T Y.
$$

Geometrically, $$\hat{Y}$$ is the projection of $$Y$$ on the space spanned by $$X_1, X_2, \ldots, X_p$$.

<img src="figures/least-squares.pdf">