I switched over here first because in the video he does Transformers and in the book they use RNNs.

We're doing the same review classification problem
![[Part 1.4 Chapter-20250621181158772.webp]]
We're going to train on top of a Wikipedia language model with IMDb info, then use that to build a classifier. ULMFit
Recall that a language model generally predicts the next word in a sentence given the previous words.

Pre-processing of text:
Tokenization: Convert text into list of word
Numericalization: Make a list of all unique words and convert each into a number.
Loader creation: fastai provides LMDataLoader for lanuage models. Creates a dependent variable offset from the independent variable by one token. Also shuffles data in a way that works with NLP
Model Creation: We need to be able to handle input lists that are arbitrarily big or small, in the book chapter we're going to use RNNs

Tokenization:
Tokenization involves some decisions that can be tricky:
Are we tokenizing each word? sub word? what about contractions? what about punctuation?
Three main approaches:
Word-based: split on spaces. Also apply language-specific rules to separate meaning (e.g. separate "don't" into "do" "n't"). Punctuation marks are separate tokens.
Subword-based: split words into smaller parts based on commonly occuring substrings. ("occasion" = "o" "c" "ca" "sion")
Character-based: split each character.

For this task we're looking at word and subword tokenization. Character based is homework.

We're not going to build our own tokenizer, we'll use someone elses.
Fastai should stay up to date with a good modern working tokenizer.

Subword tokenization can be important when we're dealing with non English languages. Japanese and Chinese for instance don't have spaces.
Hungarian and German also are able to mash words together to create complex words.
Subword tokenization is generally two steps:
1. Analyze the corpus to find most commonly occurring groups of letters
2. tokenize corpus using the vocab of subwords we've created.

When we set up a tokenizer for subword tokenization we need to decide the size of our vocabulary.
If it's large then we will have longer tokens, and less tokens to make up a sentence, faster training, less memory, less states to remember. Downsides include larger embedding matrices, which need more data to learn.
conversely if vocab size is small then the tokens are smaller and it takes more to make a sentence
Because of the flexibility of vocab size, and the vocabulary itself (can learn protein sequences or MIDI) this is likely to become the standard form of tokenization (or maybe is right now this book is old now)

For Numericalization first we need to train it by passing it our tokenized corpus.
List of lists of tokenized words.
After Numericalization you will have a list of lists of integers where the integers mp to the vocabulary.

Batching:
When we batch other types of data we make sure it's the same size.
harder to do that with text.
We need to put our data in batches that are the same length while still being sequential.
We're using a model that has memory of what it last read, so things need to be in the same order still.
The way we end up doing this is shuffling our documents at the beginning of each epoch (internal consistency per document is key) then concatenating them into one long sequence, then cutting that sequence into batches and sending them through 1 by 1.
It will always know when a knew document is starting because of the bos (beginning of stream) character.
This all happens behind the scenes in fastai when we create an LMDataLoader.

Luckily fastai will do all this preprocessing for us automatically.

With that done we can initialize our DataLoaders and our Learner, see the Jupyter notebook for details.

Now we can fine tune our language model.
it takes a long time for one epoch so we're going to fit_one_cycle. It automatically calls freeze so it will only train the embeddings with random weights, i.e. the ones not in the original vocab.

Remember to unfreeze when you want to keep learning.
We can save mid training by using learn.save(filename)
then we can load in another environment with learn.load(filename).
After the learner has been created the same way.

Once we've trained our model we can save the encoder.
This is the model not incuding the final layer.

Our ultimate goal is a review classifier, but now that we have this language model we can generate some fake reviews

Let's move on to the sentiment classifier.
When creating the datablocks we need to pass the vocab in so the indexes make sense in our embedding.
Now how are we going to minibatch these?
After tokenization these reviews still have very different lengths.
In the past we cropped - could lose valuable info
squish - how would you even do that
pad - that's the ticket!
We're going to pad with a special character that our model will ignore.

In order to improve performance we're going to sort our documents by length first and then pad the smaller ones in each set.
Each set will not be the same length. We improve efficiency by padding to the length of the largest document within the set.
This happens automatically when we use TextBlock with is_lm=False.

Once our DataBlock is set up we can make the model and then load the encoder that we saved earlier.
Now we can fine tune the model
It is smart to slowly unfreeze layers when working with NLP
computer vision it's acceptable to unfreeze all


