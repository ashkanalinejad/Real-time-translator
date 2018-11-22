THEANO_FLAGS=device=gpu,floatX=float32
python simultrans_eval.py  --id 170705-122200 --sample 1 --batchsize 1 --target 10 --sinit 1 --gamma 1 --recurrent True --Rtype 10 --coverage False 
