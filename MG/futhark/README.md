# Futhark implementation

To benchmark, run one of:

    futhark bench --backend=cuda mg.fut

    futhark bench --backend=opencl mg.fut

    futhark bench --backend=hip mg.fut

    futhark bench --backend=multicore mg.fut

(Although currently I think the dataset is just a dummy one; not the
real one.)
