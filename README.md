# How likely is a customer to buy a financial product?

Creating a complete end to end ML framework to solve binary classification problems.

The framework can be reused with other datasets by modifying code at specific points like different data wrangling steps for different datasets. Inspired by [Abhishek Thakur's video series here](https://www.youtube.com/watch?v=2wQlD46eICE&t=1062s)

The framework generically does the following
	- Takes a unclean tabular dataframe with a binary target.
	- Cleans the data (modify the code differently for different data though the function would remain the same)
	- Does a GridSearch with 10-fold CV for various sklearn models like Random Forests, Extra Trees, Gradient boosting etc..
	- Stores full CV results for analysis
	- Pick the best model and params and makes predictions on a competely unseen test set
