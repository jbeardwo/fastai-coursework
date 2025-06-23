1. What is "self-supervised learning"?
A type of learning where you don't provide correct labels.
2. What is a "language model"?
A model that takes words as input and inputs the likely next word in the sentence
3. Why is a language model considered self-supervised?
Because it provides its own labels (the next word)
4. What are self-supervised models usually used for?
Can be used on their own (autocomplete), but usually encode for other problems
5. Why do we fine-tune language models?
To specialize the model for the corpus we're using
6. What are the three steps to create a state-of-the-art text classifier?
Start with existing, train on corpus, train sentiment classifier.
7. How do the 50,000 unlabeled movie reviews help us create a better text classifier for the IMDb dataset?
By familiarizing our model with the type of language that is present in these reviews
8. What are the three steps to prepare your data for a language model?
Tokenization, Numericalization, Language model DataLoader
9. What is "tokenization"? Why do we need it?
Splits text into units that can be recognized individually. Can be more complicated than splitting on spaces
10. Name three different approaches to tokenization.
full word, partial word, character based
11. What is `xxbos`?
beginning of stream, beginning of new document
12. List four rules that fastai applies to text during tokenization.
replace_rep: replace character repeated 3 or more with xxrep, reps, char
replace_wrep: same as above but with word
spec_add_spaces: add spaces around / and #
rm_useless_spaces: remove all repetitions of space
13. Why are repeated characters replaced with a token showing the number of repetitions and the character that's repeated?
Repeated characters or words likely have a different meaning than just one. We can encode ideas about the repeated character rather than teaching it things about the singular token.
14. What is "numericalization"?
The process of assigning numbers to each token based on their position in the vocabulary
15. Why might there be words that are replaced with the "unknown word" token?
Made up words, words not present in the dataset, words left out in the vocab creation / numericalization process
16. With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset. What does the second row of that tensor contain? What does the first row of the second batch contain? (Careful—students often get this one wrong! Be sure to check your answer on the book's website.)
![[Part 1.4 Questionnaire-20250622163518756.webp]]
17. Why do we need padding for text classification? Why don't we need it for language modeling?
To make the batches the same size, squish and crop dont make sense. Not needed in LM because the documents are concatenated.
18. What does an embedding matrix for NLP contain? What is its shape?
vector representations of all tokens in the vocabulary.matrix has the size vocab size x embedding size
19. What is "perplexity"?
the exponential of the loss
20. Why do we have to pass the vocabulary of the language model to the classifier data block?
So the numericalized data lines up properly with the vocabulary.
21. What is "gradual unfreezing"?
only unfreezing the last layers and training (fine tuning), unfreezing a couple more, etc.
22. Why is text generation always likely to be ahead of automatic identification of machine-generated texts?
The classification models can be used to improve the generation algorithms (evading the classifier)