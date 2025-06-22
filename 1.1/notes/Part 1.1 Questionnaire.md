1. Do you need these for deep learning?
    
    - Lots of math T / F F
    - Lots of data T / F F
    - Lots of expensive computers T / F F
    - A PhD T / F F
2. Name five areas where deep learning is now the best in the world.
	    Medical analysis
	    computer vision
	    segmentation
	    tabulation
	    sound identification
3. What was the name of the first device that was based on the principle of the artificial neuron?
	    Mark 8 Perceptron
4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?
	    Processing units
	    Activation state
	    output function for each unit
	    pattern of connectivity
	    propagation rule 
	    activation rule combining the inputs coalescing on a unit to produce output
	    learning rule by which patterns are modified by experience
	    An environment within which the system operates.
5. What were the two theoretical misunderstandings that held back the field of neural networks?
    Basically just needed to go deeper
6. What is a GPU?
    Graphics Processing Unit
7. Open a notebook and execute a cell containing: `1+1`. What happens?
     Prints 2
8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.
    
9. Complete the Jupyter Notebook online appendix.
    
10. Why is it hard to use a traditional computer program to recognize images in a photo?
    Requires a complex set of functions that are nearly impossible to code
11. What did Samuel mean by "weight assignment"?
	    Assigning the weights that are later updated through the learning process.
12. What term do we normally use in deep learning for what Samuel called "weights"?
    Parameters
13. Draw a picture that summarizes Samuel's view of a machine learning model.
    input      ------->model
	weight   ------>  model  ------> results -----> performance
	      <----update------------------------------<
14. Why is it hard to understand why a deep learning model makes a particular prediction?
	The functions that it maps are hard to comprehend, also the weights and patterns are not necessarily aligned with logically discernable features
15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?
    universal approximation theorem
16. What do you need in order to train a model?
    data
17. How could a feedback loop impact the rollout of a predictive policing model?
    If we don't account for things like police performance/bias, we can end up sending more resources where they're not necessarily needed.
18. Do we always have to use 224×224-pixel images with the cat recognition model?
	224 is the historical standard, but ou can use pretty much any size.
19. What is the difference between classification and regression?
    Classification puts the input into one of a few discrete groups, whereas regression will find across a continuous set of real numbers
20. What is a validation set? What is a test set? Why do we need them?
	neither of these are actually used to train the model itself.
    validation set is used in the training process to check how well our model is doing
	Test set is even further removed, and is only tested on once you think the model is good. This is another way to avoid overfitting.
21. What will fastai do if you don't provide a validation set?
    split off 20% automatically
22. Can we always use a random sample for a validation set? Why or why not?
    not necessarily. Remember that the validation and test sets should be representative of the new data we'll receive and identify in the future.
	For time series data, if we're going to want to be predicting in the future it's wise to take the last bit of time we have data for as our validation set.
23. What is overfitting? Provide an example.
    Overfitting is when our model is trained to the training / dev sets too well, and doesn't generalize to other data
24. What is a metric? How does it differ from "loss"?
    A metric is a human readable measure for how well our model is doing, loss is a value that is not necessarily human interpretable, but what the model uses to model loss
25. How can pretrained models help?
    Don't need to gather data, which is the hardest part
26. What is the "head" of a model?
	The output layers responsible for making predictions. It processes the features that are calculated earlier in the model.
27. What kinds of features do the early layers of a CNN find? How about the later layers?
    Edges, gradients, lines
28. Are image models only useful for photos?
    no, other things can be represented as images and then learned (sound, files, imagination|)
29. What is an "architecture"?
    The way the model is built, size, depth, etc.
30. What is segmentation?
    Identifying pixels in an image
31. What is `y_range` used for? When do we need it?
    to specify a range when we are working with a continuous set of numbers
32. What are "hyperparameters"?
    things like learning rate, parameters that we can set before training to tweak the way the model works
33. What's the best way to avoid failures when using AI in an organization?
	Withhold data as the validation set when using external contractors. Need to avoid bias and errors