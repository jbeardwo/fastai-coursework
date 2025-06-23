We're going to talk about Natural Language Processing (NLP)
In the book we do NLP using the fastai library using Recurrant Neural Networks (RNNs)
Today we're going to use Transformers instead, and not using the fastai library.

We're going to fine tune an NLP model using a Library Hugging Face Transformers 

We're using a non fastai library even though this is a fastai course because it's good to learn more libraries. He also likes it.

HuggingFace Transformers are becoming the basis of fastai NLP

HF Transformers doesn't have the same hierarchy as fastai, so the complicated stuff isn't as abstracted.
![[Part 1.4 Video-20250621105302871.webp]]
ULMFit is a process for fine tuning an NLP model
first presented in the first fastai course, later turned into a research paper.
Step 1: Build a language model (used basically all of wikipedia) able to predict the next word given any sentence fragment.
Step 2: Took the previous model then ran a few epochs on Language model based on IMDB. So wit was good at rpedicting the next word in a movie review
Step 3: train as a sentiment classifier to predict rating based on review.

How do you go from a model that predicts the next word to a model that can be used for classification?

We're going to remove the last layer of the network that actually classifies and add another random matrix to start training.
This is really handwavey we're going to get more detailed later.

We're going to start taking a look at the Getting started with NLP for absolute beginners Kaggle Competition
![[Part 1.4 Video-20250621105857991.webp]]

US Patent Phrase to Phrase Matching competition
I believe this is a databse of patents, score can be interpreted as whether the anchor and the target are related to one another.
The context appears to represent some factor about the patent itself, which could change the way things are considered. Category
So we're trying to train a model that will output the correct score given the other info.
This can be thought of as a classification problem. 
![[Part 1.4 Video-20250621110522003.webp]]
How can we change this into a classification problem?

In jupyter notebooks you can use ! to denote a bash command, even inside of a python conditional!
Can also use {} to get data from variable i.e. ! ls {path}

We should always start a competition by looking at the dataset:
Go to the competition website and read what all the data means! Many people forget this.
Let's look at our csv files:
sample_submission.csv, text.csv, train.csv
for manipulating csv files we're going to use the pandas library.

Libraries you need to know for data science:
numpy, matplotlib, pandas, pytorch

Python for Data Aalysis is a reommeded book for pandas numpy and jupyter.

Primary data object for panda is dataframe.
We have to add context to the dataframe so it understands the input will be structured as said in the screenshot above
do this with 
` df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor

That input string isn' absolute: the delimiters help it learn, you could probably put anything there.
Trying various things is the general idea.

Before we pass to the model we need our good friends tokenization and numericalization.
Doing that depends on the model we're using.
we're going to start with a small model, microsoft/deberta-v3-small

huggingface.co/models contains a whole lot of models that are loadable with transformers.

Then we load the tokenizer from it.
Then use it to tokenize/numericalize
We need to rename 'labels' to 'score' because it usually expects labels.

Now we can create test/validation sets.

He talks about the difference between train, validation, and test.
quick review train is obvious training, validation is how we check for overfitting.
Test we don't use until we are "done" making decisions about the model and fine tuning it.

Pulling randomly from the data is not necessarily the best way.
If you want to predict the future, you should probably remove the final couple weeks to see if the model can predict what happens after what it's aware of.

If you use pictures of people (distracted driving detector example) and you have the same person in the training and validation, it may overfit to that, you may think you're doing well.
There was an instance where competition was to identify fish.
Models ended up classifying fish based on the boat because certain boats tend to catch certain fish.

cross-validation sounds pretty whack, using a rotating subset of the training set as the validation set.

Don't always trust the Kaggle split, you can customize it with great results.

What happens if you spend months training with validation and do bad on the test set?
Start over lmao.

What metric should we use to evaluate our model?
Kaggle competitions will tell you:
From patent comp:
"Submissions are evaluated on the Pearson Correlation coefficient between the predicted and actual similarity scores."

Quick review on the difference between the metric and the loss.
In real life there is no good single metric.
You should choose multiple metrics to evaluate.
Recall Precision, Accuracy, F1 score from the other course.

We're not going to look at the formula for Pearson coefficient, we're just going to get used to how it works.
it's known as r.
Varies between -1, or directly inversely correlated and 1, which is directly positively correlated


numpy has a function corrcoef that will get the correlation coefficient with every row compared to every other row

r is very sensitive to outliers in the data. Even if they look correlated on a graph it will ding significantly.
there's a square in there somewhere
'How do you decide when it's okay to get rid of outliers'
Outliers should never be just removed.
In the housing example there is probably a different type of housing/neighborhood. (low income)
Looking at the dataset, there are clearly two groups that behave differently. Probably should consider separately.
Outliers don't really exist in real life (assuming good collection), they're real things that are the result of different factors.
Make sure to investigate outliers to understand where they're oming from.


Now he's showing ways you can abuse language models
shows bots arguing about military budget
shows that more than a million anti net neutrality comments were faked

