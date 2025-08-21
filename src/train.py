import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import pickle as pkl 
import matplotlib.pyplot as plt
from pathlib import Path


def main(args):
    # Cargar dataset
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=args.seed
    )

    # Modelo
    model = RandomForestRegressor(max_depth=args.max_depth, random_state=args.seed)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log parámetros y métricas
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("seed", args.seed)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log del modelo entrenado
        mlflow.sklearn.log_model(model, "model")

        print(f"RMSE: {rmse:.3f}, R^2: {r2:.3f}")

        mlflow.sklearn.save_model(model, "models/model_ej1")

        
        # Crear gráfica
        plt.plot(y_test, preds, "o")
        plt.xlabel("Real")
        plt.ylabel("Predicción")
        plt.title("Resultados")

        # Guardar en carpeta de figuras
        fig_path = Path("reports/figures/scatter.png")
        plt.savefig(fig_path)
        plt.close()

        # Registrar como artefacto en MLflow
        mlflow.log_artifact(str(fig_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, help="Model name to log in MLflow")
    args = parser.parse_args()
    main(args)