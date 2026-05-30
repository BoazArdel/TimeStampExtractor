# examples/

Drop scanned photos here to try the tool out:

```bash
timestamp-extractor ./examples --dry-run --save-debug ./debug_out -vv
```

`--save-debug` will write the cropped timestamp region and the binarised version
for each image, so you can see exactly what the OCR pipeline saw.
