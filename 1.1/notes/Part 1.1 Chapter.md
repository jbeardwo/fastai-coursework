Overfitting is the single most important and challenging issue when training.
You will know when this is happening when our accuracy on the validation set starts going down.
FastAI will always output the validation set accuracy.

fastai defaults the validation set to 0.2, even if you forget.

Architectures with more layers take longer to train and are more prone to overfitting.
When using more data they can be a lot more accurate.

The concept of a metric for error analysis needs to be human discernable.
It is more useful as the rate of misses than the loss value, generally speaking.

vision_learner has a parameter called pretrained, which is default True, this will use the pretrained weights from the ImageNet Database.

