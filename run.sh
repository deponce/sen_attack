for model in vgg11 #resnet18 alexNet mobilenet
do
	for attack in PGD CW
	do
			python3 run.py --Model ${model} --Attack_Method ${attack} --thr 10000 --Batch_size 100

	done
done