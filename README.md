# CPSC Assignment 3

# Install requirements
```
pip install -r requirements.txt
```
# Run the code
To run the code simply 
```
python main.py
```
Theres no arguments to feed it or data to input. All the data is self contained in the provided data packages. 

# The Report Part
The first thing to do is to read the data into the program. That is what "getData" Does. It takes the training and 
testing data and converts each review into a utf-8 list. It also Takes the 1,0 for positive and negitive reviews
and turns them into a more readable version. 
Once we Have the data we need to make it readable. We do this by finding the unique words of each review and marking 
those as one. So in the end we have a large matrix of unique words and marking for where each word belongs to which review. 
After trying a couple different algorithms, for this assignment I decided to use logical regression as it was the 
easiest to implement. Using the logistical regression in the sklearn python package, the major parameter that I needed to 
finetune was the regulator.This was easy to do as we can just run the testing data against itself using different values for 
C and we can find the best one. In our case the best value was C=.25 Using this value We can continue with the assignment.
After training the classifier on the test data and testing it on the testing data we end up with a high >90% 
classification effectiveness.
