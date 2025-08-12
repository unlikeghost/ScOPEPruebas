from scope import ScOPE


sample_uk = ["ola!"]

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
    compression_metric_names=["ncd", 'clm'],
    use_prototypes=False,
    use_best_sigma=True,
    model_type='pd',
    min_size_threshold=50,
    distance_metric='euclidean',
    ensemble_strategy = 'max'
)


result = model(sample_uk, kw_samples)

print(result)

print(model)
