.ONESHELL:

train:
	python src/train.py --model RandomForest --max_depth 5 --seed 42

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns