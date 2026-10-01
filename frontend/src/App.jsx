import { useCallback, useEffect, useState } from "react";
import "./App.css";
import PredictionForm from "./components/PredictionForm";

const API_BASE_URL = "/api";

function App() {
  // =======================================================
  // STATE
  // =======================================================

  const [transactions, setTransactions] = useState([]);
  const [transactionsLoading, setTransactionsLoading] =
    useState(true);
  const [transactionsError, setTransactionsError] =
    useState("");

  const [metrics, setMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState("");

  const [modelInfo, setModelInfo] = useState(null);
  const [modelError, setModelError] = useState("");

  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");

  // =======================================================
  // API LOADERS
  // =======================================================

  const loadTransactions = useCallback(async () => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/transactions?limit=10`
      );

      if (!response.ok) {
        throw new Error(
          `Transactions request failed: HTTP ${response.status}`
        );
      }

      const data = await response.json();

      setTransactions(data.records ?? []);
      setTransactionsError("");
    } catch (error) {
      console.error(
        "Failed to load transactions:",
        error
      );

      setTransactionsError(
        "Unable to load transaction history."
      );
    } finally {
      setTransactionsLoading(false);
    }
  }, []);

  const loadMetrics = useCallback(async () => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/metrics`
      );

      if (!response.ok) {
        throw new Error(
          `Metrics request failed: HTTP ${response.status}`
        );
      }

      const data = await response.json();

      setMetrics(data);
      setMetricsError("");
    } catch (error) {
      console.error(
        "Failed to load metrics:",
        error
      );

      setMetricsError("Metrics unavailable");
    }
  }, []);

  const loadModelInfo = useCallback(async () => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/model/info`
      );

      if (!response.ok) {
        throw new Error(
          `Model request failed: HTTP ${response.status}`
        );
      }

      const data = await response.json();

      setModelInfo(data);
      setModelError("");
    } catch (error) {
      console.error(
        "Failed to load model information:",
        error
      );

      setModelError(
        "Model information unavailable"
      );
    }
  }, []);

  const loadHealth = useCallback(async () => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/health`
      );

      if (!response.ok) {
        throw new Error(
          `Health request failed: HTTP ${response.status}`
        );
      }

      const data = await response.json();

      setHealth(data);
      setHealthError("");
    } catch (error) {
      console.error(
        "Failed to load system health:",
        error
      );

      setHealthError("System unavailable");
    }
  }, []);

  // =======================================================
  // INITIAL DATA LOAD
  // =======================================================

  useEffect(() => {
    loadTransactions();
    loadMetrics();
    loadModelInfo();
    loadHealth();
  }, [
    loadTransactions,
    loadMetrics,
    loadModelInfo,
    loadHealth,
  ]);

  // =======================================================
  // REFRESH AFTER A NEW PREDICTION
  // =======================================================

  const handlePredictionComplete = async () => {
    await Promise.all([
      loadTransactions(),
      loadMetrics(),
      loadHealth(),
    ]);
  };

  // =======================================================
  // DERIVED VALUES
  // =======================================================

  const persistentFraudRate =
    metrics && metrics.database_predictions > 0
      ? (
          (metrics.database_fraud_predictions /
            metrics.database_predictions) *
          100
        ).toFixed(2)
      : "0.00";

  const systemHealthy =
    health?.status === "healthy" &&
    health?.model_loaded === true &&
    health?.database_healthy === true;

  const modelDisplayName = modelInfo?.model_name
    ? modelInfo.model_name.toUpperCase()
    : "Loading...";

  const modelVersion =
    modelInfo?.model_version ?? "—";

  const modelStage =
    modelInfo?.stage ?? "Loading...";

  const modelThreshold =
    modelInfo?.decision_threshold !== undefined
      ? Number(
          modelInfo.decision_threshold
        ).toFixed(3)
      : "—";

  const featureCount =
    modelInfo?.expected_feature_count ?? "—";

  // =======================================================
  // UI
  // =======================================================

  return (
    <div className="app">
      {/* SIDEBAR */}

      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            FD
          </div>

          <div>
            <h1>FraudGuard</h1>
            <p>Detection System</p>
          </div>
        </div>

        <nav className="navigation">
          <button
            type="button"
            className="nav-item active"
          >
            Dashboard
          </button>

          <button
            type="button"
            className="nav-item"
          >
            Predict Transaction
          </button>

          <button
            type="button"
            className="nav-item"
          >
            Transactions
          </button>

          <button
            type="button"
            className="nav-item"
          >
            Model Monitoring
          </button>
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot"></span>

          {healthError
            ? "System Unavailable"
            : systemHealthy
              ? "System Online"
              : "Checking System"}
        </div>
      </aside>

      {/* MAIN CONTENT */}

      <main className="main-content">
        {/* HEADER */}

        <header className="page-header">
          <div>
            <p className="eyebrow">
              FINANCIAL FRAUD DETECTION
            </p>

            <h2>
              Risk Intelligence Dashboard
            </h2>

            <p className="subtitle">
              Real-time transaction scoring and
              model monitoring
            </p>
          </div>

          <div className="model-badge">
            <span>Production Model</span>

            <strong>
              {modelDisplayName}{" "}
              {modelVersion !== "—"
                ? `v${modelVersion}`
                : ""}
            </strong>
          </div>
        </header>

        {/* METRICS */}

        <section className="metrics-grid">
          <article className="metric-card">
            <p>Transactions Scored</p>

            <h3>
              {metrics
                ? metrics.database_predictions
                : "—"}
            </h3>

            <span>
              {metricsError ||
                "Persisted predictions"}
            </span>
          </article>

          <article className="metric-card">
            <p>Fraud Predictions</p>

            <h3>
              {metrics
                ? metrics.database_fraud_predictions
                : "—"}
            </h3>

            <span>
              {metrics
                ? `${metrics.database_legitimate_predictions} legitimate`
                : "Loading..."}
            </span>
          </article>

          <article className="metric-card">
            <p>Fraud Rate</p>

            <h3>
              {persistentFraudRate}%
            </h3>

            <span>
              Persistent prediction history
            </span>
          </article>

          <article className="metric-card">
            <p>Decision Threshold</p>

            <h3>{modelThreshold}</h3>

            <span>
              {modelInfo
                ? `${modelDisplayName} ${modelStage}`
                : "Loading model..."}
            </span>
          </article>
        </section>

        {/* PREDICTION + MODEL STATUS */}

        <section className="dashboard-grid">
          <article className="panel prediction-panel">
            <div className="panel-header">
              <div>
                <p className="panel-label">
                  LIVE SCORING
                </p>

                <h3>
                  Transaction Prediction
                </h3>
              </div>

              <span className="api-status">
                {systemHealthy
                  ? "API connected"
                  : "Checking API"}
              </span>
            </div>

            <PredictionForm
              onPredictionComplete={
                handlePredictionComplete
              }
            />
          </article>

          {/* MODEL STATUS */}

          <article className="panel model-panel">
            <div className="panel-header">
              <div>
                <p className="panel-label">
                  MODEL STATUS
                </p>

                <h3>
                  {modelStage}
                </h3>
              </div>
            </div>

            <div className="model-details">
              <div>
                <span>Model</span>

                <strong>
                  {modelDisplayName}
                </strong>
              </div>

              <div>
                <span>Version</span>

                <strong>
                  {modelVersion}
                </strong>
              </div>

              <div>
                <span>Threshold</span>

                <strong>
                  {modelThreshold}
                </strong>
              </div>

              <div>
                <span>Features</span>

                <strong>
                  {featureCount}
                </strong>
              </div>
            </div>

            <div className="model-note">
              <span className="status-dot"></span>

              <div>
                <strong>
                  {systemHealthy
                    ? "Model service healthy"
                    : "Checking model service"}
                </strong>

                <p>
                  {modelError ||
                    healthError ||
                    "Model and database health verified through FastAPI."}
                </p>
              </div>
            </div>
          </article>
        </section>

        {/* TRANSACTIONS */}

        <section className="panel transactions-panel">
          <div className="panel-header">
            <div>
              <p className="panel-label">
                PREDICTION HISTORY
              </p>

              <h3>
                Recent Transactions
              </h3>
            </div>

            <span className="api-status">
              {health?.database_healthy
                ? "SQLite healthy"
                : "Checking database"}
            </span>
          </div>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>
                    Transaction ID
                  </th>

                  <th>
                    Prediction
                  </th>

                  <th>
                    Fraud Probability
                  </th>

                  <th>
                    Amount
                  </th>

                  <th>
                    Model
                  </th>
                </tr>
              </thead>

              <tbody>
                {transactionsLoading && (
                  <tr>
                    <td
                      colSpan="5"
                      className="empty-state"
                    >
                      Loading transactions...
                    </td>
                  </tr>
                )}

                {!transactionsLoading &&
                  transactionsError && (
                    <tr>
                      <td
                        colSpan="5"
                        className="empty-state"
                      >
                        {transactionsError}
                      </td>
                    </tr>
                  )}

                {!transactionsLoading &&
                  !transactionsError &&
                  transactions.length ===
                    0 && (
                    <tr>
                      <td
                        colSpan="5"
                        className="empty-state"
                      >
                        No prediction records
                        found.
                      </td>
                    </tr>
                  )}

                {!transactionsLoading &&
                  !transactionsError &&
                  transactions.map(
                    (transaction) => (
                      <tr
                        key={
                          transaction.transaction_id
                        }
                      >
                        <td>
                          {
                            transaction.transaction_id
                          }
                        </td>

                        <td>
                          {
                            transaction.label
                          }
                        </td>

                        <td>
                          {(
                            transaction.fraud_probability *
                            100
                          ).toFixed(4)}
                          %
                        </td>

                        <td>
                          {Number(
                            transaction.amount
                          ).toFixed(2)}
                        </td>

                        <td>
                          {
                            transaction.model_name
                          }{" "}
                          {
                            transaction.model_version
                          }
                        </td>
                      </tr>
                    )
                  )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;