from model_utils import load_model,load_baseline_stats,load_baseline_positive_rate

def feature_drift(features):
    numerical_cols = features.select_dtypes(include = ["int64","float64"]).columns
    new_stats = { }
    for cols in numerical_cols:
        new_stats[cols] = {
            "mean":features[cols].mean(),
            "std":features[cols].std()
        }

    ## Drift Decision Logic
    threshold_factor = 0.5
    drift_results = { }
    baseline_stats = load_baseline_stats()
    for cols in numerical_cols:
        mean_diff = abs(new_stats[cols]["mean"] - baseline_stats[cols]["mean"])
        threshold = threshold_factor * baseline_stats[cols]["std"]

        drift_results[cols] = {
            "mean_diff":mean_diff,
            "threshold":threshold,
            "Drift_Detected":bool(mean_diff > threshold)
        }
    return drift_results

def prediction_drift(new_data):
    pipeline = load_model()
    new_pred = pipeline.predict(new_data)
    new_positive_rate = new_pred.mean()
    train_positive_rate = load_baseline_positive_rate()
    prediction_drift = abs(new_positive_rate-train_positive_rate) > 0.1
    return {
    "baseline_rate": train_positive_rate,
    "new_rate": new_positive_rate,
    "drift_detected": bool(abs(new_positive_rate - train_positive_rate) > 0.1)
    }


def compute_status(pred_drift, feature_drift_results):
    feature_drift_detected = any(
        val["Drift_Detected"] for val in feature_drift_results.values()
    )
    if pred_drift:
        return "HIGH RISK"
    elif feature_drift_detected:
        return "MONITOR"
    else:
        return "STABLE"
