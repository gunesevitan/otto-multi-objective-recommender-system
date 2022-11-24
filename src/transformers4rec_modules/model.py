from transformers4rec import torch as tr
from transformers4rec.config import transformer
from transformers4rec.torch.ranking_metric import RecallAt


def get_model(
        schema, max_sequence_length, d_output, masking,
        transformer_model_class, transformer_model_args,
        mlp_dimensions,
):

    inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=max_sequence_length,
        d_output=d_output,
        masking=masking,
    )

    transformer_model = getattr(transformer_model_class, transformer)
    transformer_config = transformer_model.build(**transformer_model_args)

    body = tr.SequentialBlock(
        inputs,
        tr.MLPBlock(dimensions=mlp_dimensions),
        tr.TransformerBlock(transformer_config, masking=inputs.masking)
    )

    metrics = [
        RecallAt(top_ks=[20], labels_onehot=True)
    ]

    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True, hf_format=True, metrics=metrics),
        inputs=inputs,
    )

    model = tr.Model(head)
    return model
