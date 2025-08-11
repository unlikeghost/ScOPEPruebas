from scope import ScOPE


sample_uk = ["holi"]

kw_samples = {
    '2': [
        "holi",
        "hola"
    ],
    '1': [
        "adios",
        "adiox"
    ],
}


model = ScOPE(
    compressor_names=["gzip"],
    compression_metric_names=["ncd"],
    use_prototypes=True,
    use_best_sigma=True,
    model_type='ot',
    get_probas=True,
    min_size_threshold=100
)


result = model(sample_uk, kw_samples)

print(result)

print(model)
