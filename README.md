![scene.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/31-sygJAsY1LaPKIeCylh.png)

# open-scene-detection

> open-scene-detection is a vision-language encoder model fine-tuned from [`siglip2-base-patch16-512`](https://huggingface.co/google/siglip-base-patch16-512) for multi-class scene classification. It is trained to recognize and categorize natural and urban scenes using a curated visual dataset. The model uses the `SiglipForImageClassification` architecture.

```py
Classification Report:
              precision    recall  f1-score   support

   buildings     0.9755    0.9570    0.9662      2625
      forest     0.9989    0.9955    0.9972      2694
     glacier     0.9564    0.9517    0.9540      2671
    mountain     0.9540    0.9592    0.9566      2723
         sea     0.9934    0.9898    0.9916      2758
      street     0.9595    0.9819    0.9706      2874

    accuracy                         0.9728     16345
   macro avg     0.9730    0.9725    0.9727     16345
weighted avg     0.9729    0.9728    0.9728     16345
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/oqlb8a1p6zJuNZSI9PgZO.png)

---

## Label Space: 6 Classes

The model classifies an image into one of the following scenes:

```
Class 0: Buildings  
Class 1: Forest  
Class 2: Glacier  
Class 3: Mountain  
Class 4: Sea  
Class 5: Street
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio hf_xet
```

---

## Inference Code

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/open-scene-detection"  # Updated model name
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated label mapping
id2label = {
    "0": "Buildings",
    "1": "Forest",
    "2": "Glacier",
    "3": "Mountain",
    "4": "Sea",
    "5": "Street"
}

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=6, label="Scene Classification"),
    title="open-scene-detection",
    description="Upload an image to classify the scene into one of six categories: Buildings, Forest, Glacier, Mountain, Sea, or Street."
)

if __name__ == "__main__":
    iface.launch()
```

---

## Intended Use

`open-scene-detection` is designed for:

* **Scene Recognition** – Automatically classify natural and urban scenes.
* **Environmental Mapping** – Support geographic and ecological analysis from visual data.
* **Dataset Annotation** – Efficiently label large-scale image datasets by scene.
* **Visual Search and Organization** – Enable smart scene-based filtering or retrieval.
* **Autonomous Systems** – Assist navigation and perception modules with scene understanding.
