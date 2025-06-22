1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
Pictures used to train do not reflect the data used in practice.
Night photos, partially obscured photos not in training set.
Training set not gathered from the cameras that will be used
2. Where do text models currently have a major deficiency?
(feels outdated) can generate context appropriate text, but struggle with correct responses
3. What are possible negative societal implications of text generation models?
Disinformation, scams, fake news, fake friends, ethical question of companion apps
4. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
Check by hand throughout the process to help eliminate biases. Can neutralize biases along certain axes using techniques discussed in the Deep Learning Course
5. What kind of tabular data is deep learning particularly good at?
Recommendation systems
6. What's a key downside of directly using a deep learning model for recommendation systems?
They don't necessarily show what may be useful: it may show a person things they have already bought, for instance.
7. What are the steps of the Drivetrain Approach?
The Drivetrain Approach:
Define An Objective - What are we trying to do?
Levers - What inputs do we have control over?
Data - What data can we collect?
Models - How can the levers influence the objective? How do the previous questions guide our model design?
8. How do the steps of the Drivetrain Approach map to a recommendation system?
We want to increase purchases made by users
We can control what products they're shown
We know what they bought and can collect their ratings
Hopefully users who rate similar things highly will like and buy similar things
9. Create an image recognition model using data you curate, and deploy it on the web.
Done
10. What is `DataLoaders`?
Object that is used for holding that data we'll train on
11. What four things do we need to tell fastai to create `DataLoaders`?
The data we're training on, how to get it, labels, validation set parameters
12. What does the `splitter` parameter to `DataBlock` do?
defines the split for the validation set
13. How do we ensure a random split always gives the same validation set?
By choosing a seed
14. What letters are often used to signify the independent and dependent variables?
x, y
15. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
crop cuts out, pad adds black to even out, squish will push everything into fram
16. What is data augmentation? Why is it needed?
Changing the data we have slightly to increase the volume of our training set.
It can help identifying more different positions of the object in question
17. What is the difference between `item_tfms` and `batch_tfms`?
batch batches them to the GPU
18. What is a confusion matrix?
Shows how many times images from the validation/test set were mislabeled on evaluation
19. What does `export` save?
pickle file.
20. What is it called when we use a model for getting predictions, instead of training?
Inference
21. What are IPython widgets?
UI elements
22. When might you want to use CPU for deployment? When might GPU be better?
When it's what's available/cheap/low power. GPU better when you have those other things.
23. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
Latency, Security
24. What are three examples of problems that could occur when rolling out a bear warning system in practice?
False alarms, bad data, weather, slow predictions
25. What is "out-of-domain data"?
Data that doesn't match the training set
26. What is "domain shift"?
_domain shift_, where the type of data that our model sees changes over time. For instance, an insurance company may use a deep learning model as part of its pricing and risk algorithm, but over time the types of customers that the company attracts, and the types of risks they represent, may change so much that the original training data is no longer relevant.
27. What are the three steps in the deployment process?
**Manual process** – the model is run in parallel and not directly driving any actions, with humans still checking the model outputs.

**Limited scope deployment** – The model’s scope is limited and carefully supervised. 

**Gradual expansion** – The model scope is gradually increased, while good reporting systems are implemented in order to check for any significant changes to the actions taken compared to the manual process