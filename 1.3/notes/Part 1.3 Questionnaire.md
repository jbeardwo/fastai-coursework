1. How is a grayscale image represented on a computer? How about a color image?
Grayscale is an array of numbers 0-255 where 255 is black
Color has 3 channels RGB all of which go 0-255. 0,0,0 black, 255,255,255 white
2. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?
It comes with its own validation set!
3. Explain how the "pixel similarity" approach to classifying digits works.
Create ideal digit by averaging the dataset, compare input to that.
4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
`` new_list = [2*o for o in a_list if o%2==1]
5. What is a "rank-3 tensor"?
Rank is number of dimensions so you could imagine this as a cuboid
6. What is the difference between tensor rank and shape? How do you get the rank from the shape?
Rank is the length of the shape. Shape is how big each dimension is
7. What are RMSE and L1 norm?
Ways to measure the distance between points in Euclidian space. RMSE is Root Mean Square Error, is also L2 norm. L1 norm is also called mean absolute difference. 
8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
Matrix multiplication and parallelization through numpy or pytorch
9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
```
In [ ]: a = torch.Tensor(list(range(1,10))).view(3,3); print(a)
Out [ ]: tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]) 
In [ ]: b = 2*a; print(b)
Out [ ]: tensor([[ 2., 4., 6.], [ 8., 10., 12.], [14., 16., 18.]])
In [ ]: b[1:,1:] Out []: tensor([[10., 12.], [16., 18.]])
```
10. What is broadcasting?
When Numpy or Pytorch pretends a tensor is larger than it is so they can perform operations with each other.
11. Are metrics generally calculated using the training set, or the validation set? Why?
Validation set. Testing on the training material is bound to work well because it's what we trained on!
12. What is SGD?
Stochastic Gradient descent. Process of predicting, calculating loss, calculating gradients, then adjusting parameters
13. Why does SGD use mini-batches?
Doing it on the whole dataset is slower: mini batches can be parallelized.
14. What are the seven steps in SGD for machine learning?
Initialize, predict, loss, gradients, step, repeat, stop eventually
15. How do we initialize the weights in a model?
randomly
16. What is "loss"?
A measure of the error in our systems prediction.
17. Why can't we always use a high learning rate?
We will jump past the ansswer and ping-pong back and forth
18. What is a "gradient"?
The rate of change of a function with respect to certain variables in that function
19. Do you need to know how to calculate gradients yourself?
No pytorch will do this for you
20. Why can't we use accuracy as a loss function?
Loss needs to change as the weights change. A change in weights doesn't necessarily mean a change in the prediction, which could result in the accuracy not changing at all.
21. Draw the sigmoid function. What is special about its shape?
maps everything between 0 and 1
22. What is the difference between a loss function and a metric?
Loss function calculates the loss, which the system uses to evaluate its accuracy.
A metric is a human understandable measure of the systems accuracy (like literal accuracy)
23. What is the function to calculate new weights using a learning rate?
new weight = old weight - gradient*learning rate
Optimizer step is the answer they wanted
24. What does the `DataLoader` class do?
Takes a python collection and turns it into an iterator for mini batches
25. Write pseudocode showing the basic steps taken in each epoch for SGD.
predict
loss
back prop
update param as above
loop until thresh hold
26. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?
list(zip(list1,list2))
27. What does `view` do in PyTorch?
Changes the shape of a tensor without changing its contents.
28. What are the "bias" parameters in a neural network? Why do we need them?
They're like +b in y=mx+b. Without them if the input is 0 the out put will always be 0, because pretty much everything else is multiplication!
29. What does the `@` operator do in Python?
Matrix multiplication
30. What does the `backward` method do?
Backwards propagation to get our gradients
31. Why do we have to zero the gradients?
.backward will actually add the new gradients to whatever's there, so we make 0.
32. What information do we have to pass to `Learner`?
Dataloaders, model, optimization function, loss function, (optionally) metrics to print
33. Show Python or pseudocode for the basic steps of a training loop.
> ```
> def train_epoch(model, lr, params):
>     for xb,yb in dl:
>         calc_grad(xb, yb, model)
>         for p in params:
>             p.data -= p.grad*lr
>             p.grad.zero_()
> for i in range(20):
>     train_epoch(model, lr, params)
> ```
34. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.
max of 0 and the input. looks like x=y after 0, line at 0 below that
35. What is an "activation function"?
A function that has the purpose of providing non-linearity.
A series of linear layers is the same thing effectively as a single linear layer.
Adding non-linear layers in between makes this not true
36. What's the difference between `F.relu` and `nn.ReLU`?
F.relu is a python function, nn.ReLU is a pytorch module
37. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
Practical reasons:
We can use deeper models with less parameters, better performance, faster training, and less computational and memory requirements.