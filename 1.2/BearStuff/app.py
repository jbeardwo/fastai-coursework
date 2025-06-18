from fastai.vision.all import *
learn = load_learner('export.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
import gradio as gr

gr.Interface(
    fn=predict,
    inputs=gr.Image(width=512, height=512),
    outputs=gr.Label(num_top_classes=3)
).launch()

