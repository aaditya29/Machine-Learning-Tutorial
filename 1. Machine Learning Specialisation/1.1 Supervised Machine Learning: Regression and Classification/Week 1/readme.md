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

> **Goal:** The goal is to build a mathematical model that can predict the dependent variable as accurately as possible, given new values of the independent variables.<br> > **Example:** Predicting house prices (dependent variable) based on factors like square footage, number of bedrooms, location, etc. (independent variables).<br>
