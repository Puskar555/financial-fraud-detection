import { useMemo, useState } from "react";

const API_BASE_URL = "http://127.0.0.1:8001";

const FEATURE_NAMES = [
  "Time",
  ...Array.from({ length: 28 }, (_, index) => `V${index + 1}`),
  "Amount",
];

const createEmptyForm = () =>
  Object.fromEntries(FEATURE_NAMES.map((feature) => [feature, ""]));

const SAMPLE_TRANSACTION = {
  Time: 56861.0,
  V1: -1.6209258457,
  V2: 1.4084276234,
  V3: 0.873390222,
  V4: 0.1965129705,
  V5: -0.426807709,
  V6: -0.2083000417,
  V7: -0.0895107089,
  V8: 0.7673280231,
  V9: -0.4468375176,
  V10: -0.8596123977,
  V11: -0.7400106381,
  V12: 1.0152351995,
  V13: 0.880864312,
  V14: 0.2268431876,
  V15: -0.3215896291,
  V16: -0.5313055088,
  V17: 0.6048687293,
  V18: -1.0343849513,
  V19: 0.0549112897,
  V20: -0.3030692359,
  V21: 0.031090404,
  V22: -0.1073176961,
  V23: -0.0947172706,
  V24: 0.1437349691,
  V25: -0.1273035947,
  V26: 0.2115662131,
  V27: -0.5452158637,
  V28: -0.0403887145,
  Amount: 9.65,
};

function PredictionForm({ onPredictionComplete }) {
  const [formData, setFormData] = useState(createEmptyForm);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  const advancedFeatures = useMemo(
    () => FEATURE_NAMES.filter((feature) => feature.startsWith("V")),
    []
  );

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const loadSampleTransaction = () => {
    const sampleAsStrings = Object.fromEntries(
      Object.entries(SAMPLE_TRANSACTION).map(([key, value]) => [
        key,
        String(value),
      ])
    );

    setFormData(sampleAsStrings);
    setPredictionResult(null);
    setError("");
  };

  const clearForm = () => {
    setFormData(createEmptyForm());
    setPredictionResult(null);
    setError("");
  };

  const validateForm = () => {
    for (const feature of FEATURE_NAMES) {
      const rawValue = formData[feature];

      if (rawValue === "") {
        return `${feature} is required.`;
      }

      const numericValue = Number(rawValue);

      if (!Number.isFinite(numericValue)) {
        return `${feature} must be a valid number.`;
      }
    }

    if (Number(formData.Time) < 0) {
      return "Time cannot be negative.";
    }

    if (Number(formData.Amount) < 0) {
      return "Amount cannot be negative.";
    }

    return null;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setError("");
    setPredictionResult(null);

    const validationError = validateForm();

    if (validationError) {
      setError(validationError);
      return;
    }

    const payload = Object.fromEntries(
      FEATURE_NAMES.map((feature) => [
        feature,
        Number(formData[feature]),
      ])
    );

    try {
      setIsSubmitting(true);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let message = `Prediction request failed with status ${response.status}.`;

        try {
          const errorBody = await response.json();

          if (errorBody?.detail) {
            message =
              typeof errorBody.detail === "string"
                ? errorBody.detail
                : JSON.stringify(errorBody.detail);
          }
        } catch {
          // Keep the default HTTP error message.
        }

        throw new Error(message);
      }

      const result = await response.json();

      setPredictionResult(result);

      if (typeof onPredictionComplete === "function") {
        await onPredictionComplete(result);
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to complete the prediction request."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const fraudProbabilityPercent =
    predictionResult?.fraud_probability !== undefined
      ? Number(predictionResult.fraud_probability) * 100
      : null;

  const isFraud = predictionResult?.prediction === 1;

  return (
    <div className="prediction-form-container">
      <div className="prediction-form-header">
        <div>
          <p className="section-eyebrow">REAL-TIME INFERENCE</p>
          <h2>Analyze Transaction</h2>
          <p className="prediction-form-description">
            Submit the 30 model features to the production fraud detection API.
          </p>
        </div>

        <button
          type="button"
          className="sample-button"
          onClick={loadSampleTransaction}
          disabled={isSubmitting}
        >
          Load Test Transaction
        </button>
      </div>

      <form className="prediction-form" onSubmit={handleSubmit}>
        <div className="primary-feature-grid">
          <label className="prediction-field">
            <span>Time</span>
            <input
              type="number"
              step="any"
              min="0"
              name="Time"
              value={formData.Time}
              onChange={handleChange}
              placeholder="Transaction time"
            />
          </label>

          <label className="prediction-field">
            <span>Amount</span>
            <input
              type="number"
              step="any"
              min="0"
              name="Amount"
              value={formData.Amount}
              onChange={handleChange}
              placeholder="Transaction amount"
            />
          </label>
        </div>

        <div className="advanced-feature-section">
          <button
            type="button"
            className="advanced-toggle"
            onClick={() => setShowAdvanced((previous) => !previous)}
          >
            <span>Model Features V1–V28</span>
            <span>{showAdvanced ? "Hide" : "Show"}</span>
          </button>

          {showAdvanced && (
            <div className="advanced-feature-grid">
              {advancedFeatures.map((feature) => (
                <label className="prediction-field" key={feature}>
                  <span>{feature}</span>
                  <input
                    type="number"
                    step="any"
                    name={feature}
                    value={formData[feature]}
                    onChange={handleChange}
                    placeholder={feature}
                  />
                </label>
              ))}
            </div>
          )}
        </div>

        {error && (
          <div className="prediction-message prediction-error">
            {error}
          </div>
        )}

        <div className="prediction-actions">
          <button
            type="button"
            className="clear-button"
            onClick={clearForm}
            disabled={isSubmitting}
          >
            Clear
          </button>

          <button
            type="submit"
            className="predict-submit-button"
            disabled={isSubmitting}
          >
            {isSubmitting ? "Analyzing..." : "Analyze Transaction"}
          </button>
        </div>
      </form>

      {predictionResult && (
        <div
          className={`prediction-result ${
            isFraud ? "prediction-result-fraud" : "prediction-result-legitimate"
          }`}
        >
          <div className="prediction-result-heading">
            <div>
              <p className="section-eyebrow">MODEL DECISION</p>
              <h3>{predictionResult.label?.toUpperCase()}</h3>
            </div>

            <span className="prediction-result-badge">
              Prediction {predictionResult.prediction}
            </span>
          </div>

          <div className="prediction-result-grid">
            <div>
              <span>Fraud Probability</span>
              <strong>
                {fraudProbabilityPercent !== null
                  ? `${fraudProbabilityPercent.toFixed(6)}%`
                  : "—"}
              </strong>
            </div>

            <div>
              <span>Decision Threshold</span>
              <strong>
                {predictionResult.threshold !== undefined
                  ? Number(predictionResult.threshold).toFixed(3)
                  : "—"}
              </strong>
            </div>

            <div>
              <span>Model</span>
              <strong>
                {predictionResult.model_name?.toUpperCase()}{" "}
                {predictionResult.model_version}
              </strong>
            </div>

            <div>
              <span>Transaction ID</span>
              <strong className="transaction-id-value">
                {predictionResult.transaction_id}
              </strong>
            </div>
          </div>

          <p className="prediction-result-time">
            Processed:{" "}
            {predictionResult.timestamp_utc
              ? new Date(predictionResult.timestamp_utc).toLocaleString()
              : "—"}
          </p>
        </div>
      )}
    </div>
  );
}

export default PredictionForm;