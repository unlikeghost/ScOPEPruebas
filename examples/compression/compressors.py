from scope.compression import get_compressor


compressor = get_compressor("gzip", compression_level=6, min_size_threshold=0)
original, compressed, compressed_size = compressor("some string to compress")
print(compressor)
print(compressed)
print(compressed_size)

compressor = get_compressor("gzip", compression_level=6, min_size_threshold=100)
original, compressed, compressed_size1 = compressor("some string to compress")
print(compressor)
print(compressed)
print(compressed_size1)