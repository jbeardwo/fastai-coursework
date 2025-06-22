How to do a fast.ai lesson:
Watch lecture
Run notebook
Reproduce results
Repeat with a different dataset
(We did this with the llama/alpaca test for ch. 1)
(should also do for chapter 2, the bear thing, also deploy to HuggingFace and make prettier than last one. Try that DDG fix that guy suggested in discord)
There is a folder in each fastbook chapter calle clean that contains the runnable cells without text.
This can be good for verifying you know what each cell is for and predicting what it will do.

paperspace.com is another workspace notebook space that is good for this type of experimentation.
It has persistent storage and file browser and stuff


timm is a library that contains hundreds of popular architectures.
It can be good to try different architectures for your problem.
There is usually a tradeoff between speed and accuracy

List available models:
```

import timm
timm.list_models('convnext*)
```

when we instantiate our learner we need to put the architectures that are not built in fastai (i.e. any we're using from timm) in quotes, i.e.:
```

learn = vision_learner(dls, 'convnext_tiny_in22k', metrics=error+rate).to_fp16()
```

learn.predict() will output a tensor with a value for every category.
You can see the order by looking at our vocab:
```

categories = learn.dls.vocab

def classify_image(img):
	pred,idx,probs = learn.predict(img)
	return dict(zip(categories, map(float,probs)))

```
map is applying the float function to the probs iterable (which contains all the probabilities for each category unlabeled). This makes them floats.
zip pairs the two iterables, in this case it makes pairs out of the categories and the floats.
dict() just converts that into a dictionary.

If we want to look at the actual model:
```
m = learn.model
m
```
![[Part 1.3 Video-20250616111417568.webp]]
Goes on much longer but you get it.

We can get individual layers and see their weights.
The convention for locating a sublayer is in layers, see how 0 is the first layer, then model, stem, 1?
```
l = m.get_submodule('0.model.stem.1')
list(l.parameters())
```
to output the parameter weights.

![[Part 1.3 Video-20250616111506643.webp]]

Now we're going to get into how NNs work.

![[Part 1.3 Video-20250616111754064.webp]]
we're going to use a quadratic equation to visualize this concept.
Let's say that above is the function we're trying to end up at using our data.

![[Part 1.3 Video-20250616111949806.webp]]
First we're defining a generic quadratic equation.
Next we're going to use a tool called a partial evaluation. We can use this to choose fixed values for certain variables (i.e. a b and c) to create new functions.
To do this we pass the function and the variables we want to fix to the partial() function.
This is imported from functools
then we define a new function as the output of partial()
In this case we create the function f which is equivalent to $3x^2+2x+1$
With one input, that input being x.

![[Part 1.3 Video-20250616112419696.webp]]
Here he's created a dataset that roughly matches our equation.
He defines a way to make and add noise up there.
f(x) in this case is literally the function f from before applied to the dataset created by calling torch.linspace.
torch.linspace is creating an array that goes from -2 to 2 in 20 equal steps.

Let's plot a random quadratic equation over our data:
you can use the @interact tag in jupyter to make lil sliders you can use to adjust variables like so:

![[Part 1.3 Video-20250616112737442.webp]]

Now we need to define a loss function so we know if the changes are actually making it better or worse.

![[Part 1.3 Video-20250616112952715.webp]]
Mean Squared Error is simplest

![[Part 1.3 Video-20250616113044870.webp]]
We can add this as a label alongside and try our fiddling again

So we have automate this clearly
As we learned in the Deep Learning Spec this can be done quickly using the derivative.
luckily pytorch can do this for you.
![[Part 1.3 Video-20250616113410921.webp]]
making a function that makes a quadratic and evaluates its loss.
![[Part 1.3 Video-20250616113503361.webp]]
rank 1 just means 1d.
![[Part 1.3 Video-20250616113538245.webp]]
As you can see you can use a tensor directly as input to a function defined like this.
MeanBackward0 is our backprop to find grads
We can run loss.backward() to generate
This creates a new attribute for abc called grad
```
abc.grad
```
Will output the results.

We will then adjust our weights based on the gradient and go again.
![[Part 1.3 Video-20250616113902455.webp|388]]
we use with torch.no_grad() to specify what we don't want the gradients of.

Automate this shit:
![[Part 1.3 Video-20250616114046090.webp|452]]

Now he's defining ReLu:
![[Part 1.3 Video-20250616114240290.webp|292]]
don't forget about @interact
![[Part 1.3 Video-20250616114315227.webp|322]]

DOUBLE RELU????

![[Part 1.3 Video-20250616114353658.webp]]
![[Part 1.3 Video-20250616114408558.webp]]
These are important because they don't necessarily effect the relus behind them.
You could construct pretty much any curve by putting ReLu's together.
Even something like human speech could be approximated.
Obviously this lets us make very specific custom functions.

Jeremy almost always starts a new project with resnet18 so that he can try as many different things as possible.
Trying different architectures is the LAST thing.
Do we need faster? more Accurate? much easier to make this decision if we've exhausted all our other options.

Now he's asking why we don't change it by the gradient instead of scaling it.
apparently as you get very close to the correct value it always looks like a quadratic.

The reason we scale the gradient is that we can move past the correct value, ping ponging back and forth past it.

That's why we want a smaller learning rate.
Again we learned this in the Deep Learning Course.
Too small takes too long, too big creates pingponging, may never settle.


We're gonna need to calculate a lot of ReLU's.
We're going to use Matrix Multiplication and he's gonna talk about that for  bit.
matrixmultiplication.xyz can show this visually.


Now we're going to try to learn who is more likely to survive Titanic:

![[Part 1.3 Video-20250616122514459.webp]]
Notice how we're taking certain characteristics and representing them as binary (1 for male, 0 for female) or one-hotting them in the case of the embark letter.

WERE NOT ACTUALLY ONE HOTTING
notice that Pclass on the left has three values, we only have Pclass_1 and Pclass_2 on the right because if both of those are 0 then it must be Pclass 3.
This is a technique for encoding dummy variables.

![[Part 1.3 Video-20250616122658961.webp]]
We randomly initialize our parameter values.
In this case the fare value from before is too much bigger than everything else, it outinfluences every other variable. We'd like all variables to be normalized between 0 and 1 to prevent this.

Same idea with age, we're going to divide the age by the maximum.
He's going to scale the fare as follows:
notice that there are very small numbers and relatively larger numbers
In this case a log works really well, we'll add 1 to get rid of negative values:
![[Part 1.3 Video-20250616123047986.webp]]
It can still be greater than 1, but it's not enough to be an issue.
![[Part 1.3 Video-20250616123637648.webp]]
Just showing some excel context for sum-product our Params and our data
Notice the column at the end called "Ones".
He's doing this sort of like the b parameter, learning the const term we see on the right.
![[Part 1.3 Video-20250616123956585.webp|163]]
He runs it and gets this.
Wants to bind it between 0 and 1, moves on before elaborating.

Now he's doing this multiple times to represent multiple neurons in a network.
Make sure you transpose your parm matrix relative to what was  shown above if you want to do Matrix multiplication (MMULT) rather than than SumProduct



He jumps ahead to start talking about NLP and the Getting started with NLP notebook, this is actually covered in the next section I'm pretty sure.
We're going to use the Hugging Face Transformers library for this assignment.
It's good to be comfortable with more than one library, he also says this one is very good.

Oh okay he's saying to go through this before the next lesson "if you have time"