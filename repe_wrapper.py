
import torch

from transformers import pipeline

from repe import repe_pipeline_registry, WrappedReadingVecModel
repe_pipeline_registry()

import random
random.seed(42)

# class WrappedModel():
#     def __init__(self, model, tokenizer, device):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device
    
def get_wrapped_model(model, tokenizer, pn_pairs, layer_id, coeff, batch_size=16):


    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    train_labels = []
    for d in pn_pairs:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])

    rep_reader = rep_reading_pipeline.get_directions(
        pn_pairs, 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=train_labels, 
        direction_method=direction_method,
        batch_size=batch_size,
    )
    activations = {}
    for layer in layer_id:
        activation = torch.tensor(coeff * rep_reader.directions[layer][0] * rep_reader.direction_signs[layer][0], dtype=model.dtype).to(model.device)
        
        activations[layer] = activation

    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    wrapped_model.unwrap()
    wrapped_model.wrap_block(layer_id, block_name="decoder_block")

    ### Controlled model hidden_states:
    wrapped_model.set_controller(layer_id, activations, masks=1)
    
    return wrapped_model
