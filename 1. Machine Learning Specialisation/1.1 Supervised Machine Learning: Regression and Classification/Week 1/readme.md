# Important Notes Week 1

## Supervised VS. Unsupervised Machine Learning

#### 1. What is Machine Learning?<br>

Machine learning is a branch of artificial intelligence (AI) that focuses on the use of data and algorithms to enable computers to learn and improve their performance over time. It allows computers to make predictions, classify information, cluster data, and more, by finding relationships and patterns in data.<br>

#### 2. Supervised Learning:

Imagine you're teaching a computer to recognize different types of fruits. You gather a bunch of examples where each fruit is labeled (like apples, oranges, and bananas).

- **Data Collection**: You show the computer pictures of apples, oranges, and bananas. Each picture comes with a label, telling the computer what fruit it is.
- **Training**: The computer learns by looking at these pictures and labels. It figures out patterns and features that distinguish apples from oranges and bananas.
- **Testing**: Once the computer thinks it has learned enough, you give it new pictures of fruits it has never seen. You check how well it can correctly identify these fruits based on what it learned during training.
- **Feedback and Improvement**: If it makes mistakes, you correct it, and it learns from those mistakes. You repeat this process until the computer gets really good at recognizing different fruits.
- **Prediction**: Now, when you show it a new picture of a fruit, it should be able to tell you whether it's an apple, orange, or banana based on what it learned.

So, in a nutshell, supervised learning is like teaching a computer to make predictions or identify things by showing it examples and providing feedback until it gets really good at it. It's a bit like how we learn by being shown examples and receiving feedback on what's right or wrong.

#### 3. Unsupervised Learning:

Imagine you have a bunch of different fruits, but this time, you don't have labels telling you what each fruit is. You just have a mix of apples, oranges, and bananas.

- **Sorting without Labels**: Unsupervised learning is like trying to sort these fruits into groups without any hints or labels. You don't know which ones are apples, oranges, or bananas initially.

- **Finding Patterns on Its Own**: The computer starts looking at the fruits and tries to find patterns or similarities. It might notice that some fruits are round, some are elongated, and others have different colors.

- **Grouping Similar Things Together**: Based on these patterns, the computer groups similar-looking fruits together. It doesn't know their names, but it sees that certain fruits share common characteristics.

- **Clustering**: This process is called clustering. The computer is essentially saying, "These fruits look alike, so they must belong to the same group." It forms clusters without being told what each cluster represents.

- **Discovering Structure**: After forming these groups, you might notice that within each group, the fruits have something in common. Maybe one cluster has mostly red fruits, another has mostly yellow fruits, and so on.

In a nutshell, unsupervised learning is like letting the computer explore and find its own patterns in data without being told what to look for. It's great for discovering hidden structures or relationships in information, even when you don't have predefined categories.

## Regression Model

**What is Regression?**<br>

- **Core Idea:** Regression in machine learning involves finding the relationship between a dependent variable (the thing you want to predict) and one or more independent variables (the factors influencing the prediction).<br>

Imagine regression in machine learning is like playing detective with numbers:<br>

**The Mystery**<br>
You want to figure out something, like how much ice cream you might sell on a hot day. That's the mystery you're trying to solve!
You have clues:

- The temperature outside
- If it's a weekend or a weekday
- Maybe even how many flavors you have that day

**The Detective (Regression Model)**<br>
A regression model is like a super-smart number detective. It looks at all the clues from past days (when you sold ice cream, how hot it was, etc.) It tries to find a secret pattern that connects the clues to how much ice cream you sold.

**The Secret Pattern**<br>
Maybe our detective sees that the hotter it is, the more ice cream you sell. That's a pattern!<br>
Sometimes the pattern is a simple line, like on a graph. Other times, it might be a bit curvier. The detective figures that out.

**Solving New Mysteries**<br>
Once the detective knows the pattern, it's prediction time!
You can ask, "If tomorrow is 90 degrees, how many ice creams might I sell?"
The detective uses the pattern to make its best guess!<br>

> **Goal:** The goal is to build a mathematical model that can predict the dependent variable as accurately as possible, given new values of the independent variables. <b>For example:</b> Predicting house prices (dependent variable) based on factors like square footage, number of bedrooms, location, etc. (independent variables).<br>

#### Some Terminologies

- Training Set: Data used to train the model set.
- `x` = "input" variable feature
- `y` = "output/target" variable feature
- `m` = number of training examples
- `(x,y)` = single training example
- (x<sup>(i)</sup>, y<sup>(i)</sup>) = i<sup>th</sup> training example

### Single Variable Linear Regression Model

Univariate linear regression seeks to find the best straight line that describes the relationship between a single independent variable (X, the predictor) and a single dependent variable (Y, what you're trying to predict).<br>

#### How It Works

1. Data Collection: You gather data containing paired observations of your independent and dependent variables.
2. Model Fitting: The model uses an algorithm (often "least squares") to find the best values that minimize the overall error (the difference between the predicted values and the actual values in your data).
3. Making Predictions: Once you have your equation, you can plug in a new value of X and the model will predict the corresponding value of Y.

#### Key Concepts

1. Linearity: The model assumes a linear relationship between the variables. A scatter plot can help you visualize if this assumption is reasonable.
2. Least Squares: The most common method for fitting the model. It aims to minimize the sum of the squared errors.
3. Assumptions: For reliable results, your data should generally meet certain assumptions such as normally distributed errors.

**Imagine a Graph**<br>
Think of a graph with dots scattered across it. Univariate linear regression is all about drawing the best line through those dots.

- **Univariate:** This means we have one special "thing" we want to predict (like the price of a house) and one clue to help us (like the house's square footage).
- **Linear:** This means we're trying to find a straight line fit for the data.
- **Regression:** This is the fancy word for finding that best-fit line to make predictions.

**The Detective's Line**<br>
Imagine those dots on the graph represent houses:

- Each dot is a house.
- The position on the x-axis (horizontal) shows how big the house is (square footage).
- The position on the y-axis (vertical) shows how much the house sold for.

Our detective line (the regression model) tries to go through the middle of the dots as best it can.

**Why a Line?**<br>

The line helps us predict! Let's say you find a new house on the market, and you know its size.

1. Find the size on the x-axis of our graph.
2. Go straight up until you hit the detective line.
3. Move across to the y-axis – that's roughly how much the house might cost based on its size!

**The Math Part (Simplified):**

The detective line is actually an equation:<br>
`Price = (Slope * Square Footage) + Intercept`

- Slope: How steep the line is. A steeper line means the price changes more for each extra bit of square footage.
- Intercept: Where the line crosses the y-axis (like the starting price, even for a tiny house).

**Real-World Examples:**<br>

Univariate linear regression isn't just about houses but we can use it to:

- Predict someone's height based on their age (when they're young)
- Estimate a student's test score based on the hours they studied.
- Predict how long a journey takes based on the distance traveled.

### Cost Function(Squared Error Cost Function And Its Formula)

The squared error cost function, also known as the Mean Squared Error (MSE), is a workhorse in linear regression. This function squares the errors which gives higher weight to larger errors.<br>

> The MSE formula is:

$$
MSE = \frac{1}{2m}\sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

<br>

Here's a breakdown of the symbols:

- **MSE:** Mean Squared Error (the cost function we're calculating).
- **m:** Total number of data points in our dataset.
- **Σ:** Summation symbol (tells us to sum the following terms for all data points).
- **y_cap:** The actual value for a data point
- **y:** The value predicted by our model for that data point.
- **^2:** Represents squaring the term inside the parenthesis.

#### Understanding The Formula

Imagine you're playing a guessing game where you try to predict how many candies are in a jar. You guess different numbers, and someone tells you how far off you are from the real number.

The squared error cost function is like a tool to help you win this guessing game. It measures how wrong your guesses are on average. Here's how it works:

1. **Guess vs. Reality:** For each guess you make (let's call it your "prediction"), you find the difference between your guess and the actual number of candies (the "real" number).
2. **Squaring the Mistake:** Since some guesses might be a little under and some a little over, we don't want to consider negative mistakes. So, we square the difference. Squaring a number makes it positive, even if it was negative before. The bigger the difference, the bigger the square will be.
3. **Adding it Up:** We don't just care about one guess; we want to see how well we're doing overall. So, we add up the squared mistakes for all your guesses.
4. **Finding the Average:** To make things fair, especially if you made a lot of guesses, we take the total of squared mistakes and divide it by the number of guesses you made. This gives us the average amount you were wrong by squaring.<br>

The lower this squared error cost function is, the closer our guesses are to the real number of candies on average! This helps us adjust our guesses and get better at predicting how many candies are there.<br>

#### What Does MSE Tell Us?

A lower MSE indicates a better fit for your model. It means the average squared difference between the actual values and predicted values is smaller, signifying the predictions are closer to the real values.<br>
A higher MSE suggests a poorer fit. The model's predictions are deviating more from the actual values.

#### Minimizing MSE

The goal of training a linear regression model is to find the model parameters (slope and intercept of the fitting line) that minimize the MSE. Algorithms like gradient descent use the MSE to iteratively adjust these parameters, guiding the model towards better predictions.

In essence, the squared error cost function helps us quantify the overall discrepancy between a model's predictions and the actual data points.

## Training The Model with Gradient Descent

### What is Gradient Descent?

- **Optimization Algorithm:** Gradient descent is a popular first-order iterative algorithm used to find the minimum of a function. It's a core concept behind training many machine learning models.
- **Intuition:** Imagine you're lost in hilly terrain and want to find the lowest point. Gradient descent is like methodically feeling the slope around you with your feet and taking steps in the direction of the steepest descent. Eventually, you'll end up in a valley (hopefully the lowest one!).

### How It Works

1. **Start with Random Values:** You begin by initializing the parameters (weights and biases in machine learning) of your model to some random values.

2. **Calculate the Gradient:** The gradient is a vector that points in the direction of the steepest increase of a function and has a magnitude representing how steep the change is. In gradient descent, you calculate the gradient of your cost function (a measure of how 'wrong' your model currently is) with respect to the parameters.

3. **Update Parameters:** You adjust the parameters in the opposite direction of the gradient, proportional to the gradient's magnitude. The idea is that moving against the direction of the steepest increase will lead you toward a minimum.

4. **The Learning Rate:** The learning rate is a crucial hyperparameter (a setting you decide) that controls the size of the steps you take during updates. A large learning rate could lead to overshooting the minimum, while a too-small learning rate might cause very slow convergence.

5. **Repeat:** Steps 2-4 are iterated until you reach a point where the gradient is close to zero, indicating you've found a minimum (at least a local one).

### Layman Term Explanation

Let's imagine gradient descent is a treasure hunt game:<br>

#### The Treasure (the solution)

You want to find a buried treasure at the bottom of a big, bumpy hill. This treasure is like the best answer or solution to a problem.

#### The Blindfold (starting with guesses)

You start the game wearing a blindfold. You don't know where the treasure is, so you just pick a random spot on the hill to start. This is like starting with random guesses for your solution.

#### Feeling Around (the gradient)

You can feel the slope of the ground around you. If it's going uphill, you know the treasure isn't in that direction. If it's going downhill, that's a good sign! The slope is like the gradient, it tells you which direction gets you closer to the goal.

#### Tiny Steps Downhill (updating your guess)

You carefully take small steps in the downhill direction. You keep checking the slope after each step to make sure you're still heading the right way. These small steps are like changing your guess to make it better and closer to the real solution.

#### The Learning Rate (how big are your steps?)

How big your steps are is called the learning rate. Big steps get you downhill faster, but you might jump right over the treasure! Small steps take more time, but you're less likely to miss the spot.

#### Finding the X (the minimum)

Eventually, you'll reach a flatter part of the hill. The slope won't change much in any direction. You've found the 'X' on the treasure map - you're close to your goal!

#### Gradient Descent in Computers

Computers use something similar to find the best solution to problems:<br>

- The hill is a way to picture how 'wrong' a solution is. Lower means better.
- Instead of feeling the ground, computers calculate the gradient using math.
- They keep taking small steps guided by the gradient until they find the lowest spot.

### Machine Learning Application

Gradient descent is the backbone of training various machine learning models:

- **Linear Regression:** Finding the best-fit line by minimizing the mean squared error.
- **Logistic Regression:** Finding the optimal decision boundary for classification tasks.
- **Neural Networks:** The heart of the backpropagation algorithm used to train complex neural networks.

### Important Considerations

- **Local Minima:** Gradient descent is prone to finding local minima, not necessarily the global minimum. Techniques like random restarts, momentum, or adaptive learning rates can help mitigate this issue.
- **Choosing a Learning Rate:** The learning rate is highly influential and requires careful tuning.
- **Feature Scaling:** Standardizing or normalizing your data can improve the convergence speed of gradient descent.

#### General Gradient Descent Formula

$$
GradientDescent = \theta_{t+1} = \theta_t - \eta \nabla_{\theta} J(\theta_t)
$$

**Explanation:**<br>

- $\theta_{t+1}$ : UpdatedParameters (weights or biases at the next time step.

- $\theta_t$ : Current parameters.
- $\eta$ : Learning rate (symbol eta). This controls the size of the steps taken.
- $\nabla_{\theta} J(\theta_t):$ Gradient of the cost function (J) with respect to the parameters $(\theta)$ evaluated at the current parameters.

#### Important Notes

- This formula represents the core idea of gradient descent. There are variations like Stochastic Gradient Descent (SGD) and Mini-batch Gradient Descent that change how the gradient is calculated.
- The cost function (J) will be different depending on the type of machine learning model you're training.

### The Role of the Learning Rate

The learning rate is a crucial hyperparameter (a value we set before training). It determines how large the steps are that the algorithm takes during its descent. Here's how it impacts things:

- **Too Small:** If the learning rate is too small, the algorithm will take tiny steps. Convergence towards the minimum will be slow, and the model risks getting stuck in shallow local minima (small dips, but not the true lowest point).

- **Too Large:** If the learning rate is too large, the algorithm may take giant leaps and recklessly overshoot the minimum. It might even diverge, becoming unstable and failing to find any solution at all.

- **Just Right:** A good learning rate helps find a balance. Steps are big enough to make progress, but controlled enough.

#### Impact on Gradient Descent:

1. **Convergence:**

   - Too high a learning rate can make the updates too large, leading to oscillations around the minimum instead of finding it. The algorithm might even diverge, getting worse with each update!
   - Too small a learning rate makes your "walk" painfully slow, drastically increasing the time to find a good solution.

2. **Finding True Minimum:**

   - A very high learning rate might cause you to jump right past little dips and valleys that contain the actual best solution (global minimum).
   - Smaller learning rates let you explore the space more thoroughly, increasing the chance of finding the true best solution.
