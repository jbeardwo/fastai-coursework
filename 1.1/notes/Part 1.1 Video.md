Tweaking Neural networks isn't much of a thing in practice.
There are a few models that the community has figured out cover pretty much all use cases.
fastai is a python library build on top of pytorch

DataBlock is a object used for training and validation, it has some prarameters:
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

blocks are the types of input, output.
in this case Images go in, and Categories go out.

get_items defines a function for retrieving the data
in this case it's going to get a list of image files.

splitter defines the split of your validation set data
In this case we're going to use twenty percent of the data

get_y defines a function for defining the correct output of the training set.
In this case parent_label just returns the parent folder name of the image.

Item transforms is going to resize our input so that they can be the same size.
There's 2 ways to resize: crop out a piece in the middle, or squish. We're squishing.

dataloaders are a class that pytorch iterates through that inputs the data parallel
Feeds batches to GPU

dls.show_batch(max_n=6)
This will show 6 examples from the set.

docs.fast.ai has tutorials and API information for all these functions and objects.

A learner is something that combines a model and the training data.
We pass it the data and the model:

learn = vision_learing(dls, resnet18, metrics=error)
There is information about the models on fast.ai documentation as well.
rwightman.github.io/pytorch-image-model/results/

fine_tune() automatically uses best practices for fine tuning a pre-trained model:
learn.fine_tune(3)

we can then make a prediction
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")

This isn't just for image recognition, for example:

##### Segmentation:
![[Part 1.1-20250607132157826.webp|231]]
This was trained on a very small dataset in less than two minutes
left side is input which was categorized by hand (people are ground, yikes)
Output on the right is pretty good considering that.
He says it could be like perfect if he trained it a little longer.

![[Part 1.1-20250607132434686.webp]]
That is the code for the above model, it is pretty similar to what we did before.
This time we're using a particular DataLoader for Segmentations.
These exist for many common used datatypes, are generally better tuned for their function and allow you to do less.


##### Tabular analysis:
taking things like spreadsheets and trying to make columns out of the inputs
The code for this is similar to what we've done already.
![[Part 1.1-20250607132656304.webp]]

we're grabbing some data using untar_data, which is a fastai function for decompressing, then we call another type of DataLoader for Tabular.
cat_names are categorical entries, so one of a few choices
cont_names are continuous, meaning they can be any real number.

![[Part 1.1-20250607133655763.webp]]
we're going to use fit instead of fine tune because all tabular data is quite different.



##### Collaborative filtering - recommendation system
![[Part 1.1-20250607133805887.webp]]
![[Part 1.1-20250607133919786.webp|526]]

because it's predicting a real number we'll give it a range.
The actual range is 1 to 5 but we'll learn later why we go a little past that.

