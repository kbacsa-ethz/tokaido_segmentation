# Change data path accordingly
DATADIR="/app/data"
python train.py --backbone resnet50 --root-path $PWD --data-path "$DATADIR" --device cuda --batch-size 8 --num-workers 8 --n-epochs 300 --lmdb
MODELDIR=$(ls -td $PWD/runs/* | head -1)
MODELDIR="$MODELDIR/best_model.pth"
echo "Running evaliation and test on $MODELDIR"
python eval_model.py --backbone resnet50 --root-path $PWD --data-path "$DATADIR" --model-path "$MODELDIR" --device cuda
python test_model.py --backbone resnet50 --root-path $PWD --data-path "$DATADIR" --model-path "$MODELDIR" --device cuda
