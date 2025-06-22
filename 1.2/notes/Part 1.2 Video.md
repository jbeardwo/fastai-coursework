How do we put our model into production?
First we need data, next we need to clean the data.

ims = search_images_ddg('search term')

Data cleaning:
Weirdly enough, we're going to train the model first, then clean the data.
He shows some images showing the differences between crop, pad, and squish.
Pad pads with black
crop crops out the image
squish captures most of the data but squishes it
We can use randomresized crop
bears - bears.new(item_tfms=RanedomResizedCrop(128, min_scale=0.3))
to create many examples from one large image. We'll also get slightly different images per epoch.
There's also aug_transforms that are a litte crazier.


Generally speaking if you're training more than like 5 or 10 epochs (usually you are, unless the problem is very easy) it's probably a good idea to used Random Resized Crop and Aug Transforms
![[Part 1.2 Video-20250608142132249.webp]]

![[Part 1.2 Video-20250608142249629.webp]]

![[Part 1.2 Video-20250608142413560.webp]]
You can use the plot confusion matrix to view the misclassified images.

![[Part 1.2 Video-20250608142458989.webp]]
shows the images that are "most wrong"
Sometimes we'll get the correct answer with bad loss, this is because the model wasn't very confident about its evaluation.

![[Part 1.2 Video-20250608142601744.webp]]
Now we can use the fastai Image Classifier Cleaner 

![[Part 1.2 Video-20250608143022885.webp]]
You can choose which class and which set to choose from.
Then it will show you the items with the highest loss from that set.
You can by hand reassign it to the correct category, or delete (like this dog here)
![[Part 1.2 Video-20250608143234183.webp]]
These two lines will then delete and reassign the values, respectively.

Always run your model before cleaning, it will help you find out what parts of the problem are hard as well as what the next steps are, gathering more data, pre-cleaning, can we clean in an automated way?

huggingface spaces are where we're going to serve our model.
Use apache 2.0 to avoid patent issues lol
Use gradio as your SDK
We're going to use git to edit this space.
It will give you the code to clone the repository as well as the code for your first file called applpy.

One you commit it the site will update.
By default there is a text box labeled name, output adds hello before.

After we create and train our learner we're going to use
learn.export('model.pkl') 
pickle lol
In Kaggle anything you save will show up in the data tab.
we're now going to download the .pkl file and add it to our repository.

How do we do predictions on our saved model?
![[Part 1.2 Video-20250608150127043.webp]]
Any external functions used in your labeling need to be included because the learner requires that data. (is_cat above.)

im is our representation of the image file after processing.

learn = load_learner('model.pkl')
learn.predict(im)

Now we want to make a gradio interface that contains this info.
Gradio requires us to give a function
![[Part 1.2 Video-20250608150418438.webp]]

So to create the interface we tell it what function, what input, what output, and we can provide samples
![[Part 1.2 Video-20250608150557927.webp]]

When we run this in Jupyter it is actually running locally, make sure to stop it.

Now we need this to all be contained in a python script.
He typed #| export at the top of each cell that has info he wants in his script.
Then at the bottom he imports
from nbdev.export import notebook2script
notebook2script('app.ipynb')
Passing the name of this notebook creates app.py containing that script.
This way you can do experimentation in the notebook and copy all changes automatically.
Can avoid confusion forgetting to update and also saves clicks.
