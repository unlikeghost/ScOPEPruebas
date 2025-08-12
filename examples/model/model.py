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
    compressor_names=["gzip", "bz2"],
    compression_metric_names=["ncd", "clm"],
    use_best_sigma=True,
    model_type='ot',
    min_size_threshold=100,
    matching_metric="jaccard",
    aggregation_method='median'
)


result = model(sample_uk, kw_samples)

print(result)

print(model)