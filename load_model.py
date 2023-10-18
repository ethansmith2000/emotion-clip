import torch
import transformers

from transformers import BertForSequenceClassification
import os
from train import Network
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from network import preprocess_image
from transformers import CLIPVisionModelWithProjection

# "model_name_or_path": "bert-base-uncased",
#"model_name_or_path": "xlm-roberta-base",
#"model_name_or_path": "xlm-roberta-large",


# a wrapper for inference
class EmotionClassifier(torch.nn.Module):
    def __init__(self, vision_model, classifier, num_cutouts=4, fit_method="pad"):
        super().__init__()
        self.vision_model = vision_model
        self.classifier = classifier
        self.num_cutouts = num_cutouts
        self.fit_method = fit_method
        self.preprocess = preprocess_image
        self.is_clip = "clip" in vision_model.__class__.__name__.lower()

    def forward(self, image, skip_preprocess=False):
        with torch.no_grad():
            # if not isinstance(image, list):
            #     image = [image]
            # image = [im.convert("RGB") for im in image]

            image = self.preprocess(image, num_cutouts=self.num_cutouts, fit_method=self.fit_method, is_clip=self.is_clip).to(self.vision_model.device).to(self.vision_model.dtype)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            output = self.vision_model(image).last_hidden_state

            if self.num_cutouts > 0 and self.fit_method == "sliding":
                output = output.reshape(-1, self.num_cutouts, *output.shape[-2:])
                kernel = gaussian(self.num_cutouts, 1)
                kernel = kernel / kernel.sum()
                output = (output * kernel.reshape(-1, 1, 1, 1)).sum(dim=1)

            logits = self.classifier(output).squeeze()
            output = self.sigmoid(logits)

        return output


def load_model(model_path, device="cuda", dtype=torch.float16):
    state_dict = torch.load(model_path)

    model = CLIPVisionModelWithProjection.from_pretrained(state_dict["base_model_name"])

    hidden_dim = state_dict["classifier"]['classifier.weight'].shape[0]
    if "linear.weight" in state_dict["classifier"].keys():
        input_dim = state_dict["classifier"]['linear.weight'].shape[1]
    else:
        input_dim = state_dict["classifier"]['classifier.weight'].shape[1]

    classifier = Network(in_dim=input_dim,out_dim=hidden_dim, act_fn=state_dict["act_fn"]).to(device).to(dtype)
    classifier.load_state_dict(state_dict["classifier"])


    model = EmotionClassifier(model, classifier, num_cutouts=state_dict["num_cutouts"], fit_method=state_dict["fit_method"])
    model = model.to(device).to(dtype)

    return model