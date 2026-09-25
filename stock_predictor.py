import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():

    print("=" * 50)
    print("          STOCK PRICE PREDICTOR")
    print("=" * 50)

    # Get stock ticker from user
    ticker = input(
        "\nEnter stock ticker (Example: AAPL, MSFT, GOOGL): "
    ).strip().upper()

    # Use AAPL if user enters nothing
    if not ticker:
        ticker = "AAPL"

    print(f"\nDownloading historical data for {ticker}...")

    # Download historical stock data
    try:
        data = yf.download(
            ticker,
            start="2020-01-01",
            end="2026-01-01",
            auto_adjust=False,
            progress=False
        )

    except Exception as error:
        print("\nError downloading stock data:")
        print(error)
        return

    # Check whether data was downloaded
    if data.empty:
        print(
            f"\nNo data found for '{ticker}'. "
            "Please check the ticker symbol."
        )
        return

    # Fix yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Keep required columns
    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    data = data[required_columns].copy()

    # Create next-day closing price target
    data["Next_Close"] = data["Close"].shift(-1)

    # Remove rows containing missing values
    data = data.dropna()

    # Make sure enough data exists
    if len(data) < 100:
        print("\nNot enough historical data to train the model.")
        return

    print(f"\nTotal usable records: {len(data)}")

    # Features used for prediction
    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    X = data[features]
    y = data["Next_Close"]

    # Chronological 80/20 split
    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")

    print("\nTraining period:")
    print(
        X_train.index.min().date(),
        "to",
        X_train.index.max().date()
    )

    print("\nTesting period:")
    print(
        X_test.index.min().date(),
        "to",
        X_test.index.max().date()
    )

    # Create and train Linear Regression model
    model = LinearRegression()

    model.fit(X_train, y_train)

    print("\nModel trained successfully!")

    # Predict test data
    predictions = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # Create results DataFrame
    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    })

    print("\nSample Predictions:")
    print(results.head(10))

    # Save all test predictions
    results_file = f"{ticker}_predictions.csv"

    results.to_csv(results_file)

    print(
        f"\nPredictions saved as {results_file}"
    )

    # Create graph
    plt.figure(figsize=(12, 6))

    plt.plot(
        y_test.index,
        y_test.values,
        label="Actual Price"
    )

    plt.plot(
        y_test.index,
        predictions,
        label="Predicted Price"
    )

    plt.title(
        f"{ticker} Actual vs Predicted "
        "Next-Day Closing Price"
    )

    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")

    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save ticker-specific graph
    graph_file = f"{ticker}_prediction_graph.png"

    plt.savefig(
        graph_file,
        dpi=300
    )

    print(
        f"Graph saved as {graph_file}"
    )

    plt.show()

    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()