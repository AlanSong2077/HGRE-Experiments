import gzip, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
    while True:
        chunk = fi.read(1 << 20)
        if not chunk: break
        fo.write(chunk)
print(dst)
