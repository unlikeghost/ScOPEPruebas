from scope.compression import CompressionMatrixFactory


test_samples = {
    0: ['Hola', 'Adios', 'Buenos dias'],
    1: ['Hello', 'Goodbye', 'Good morning']
}

test_sample = 'hola'
    
matrix = CompressionMatrixFactory(
    compressor_names=['gzip','bz2'],
    compression_metric_names=['ncd', 'clm'],
    concat_value=" ",
    min_size_threshold=80
)

data = matrix.build_matrix(
    sample=test_sample,
    kw_samples=test_samples,
    get_sigma=True
)
    
print(data)