## Train Instructions


### BLIS: ###

```
python train_blis.py --train_results_path train/ --model_path models/
```

### Queueing Model ###

```
python train_qm.py --train_results_path train/
```

This gives you files (one `QM_train.json` per TP folder under `train/`) to feed into the Go binary for `model-trainer`.
...

## Test Instructions

### BLIS: ###

```
python test_blis.py --test_results_path test/ --model_path models/ --groupby_field tp/chunk_size/app/rps
```

### Queueing Model ###

```
python test_qm.py --test_results_path test/
```

This gives you a combined file `QM_test_TP=2/4/8.json` to feed into the Go binary.