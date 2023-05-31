from transformers import AutoFeatureExtractor, ASTForAudioClassification

def load_AST():
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model.config.output_hidden_states = True
    for param in model.parameters():
        param.requires_grad = False
    return feature_extractor, model
