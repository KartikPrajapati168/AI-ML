import { useState } from "react";
import API from "../services/api";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from "recharts";

function Prediction() {

    const [formData, setFormData] = useState({
        age: "",
        gender: "Male",
        tenure: "",
        usage_frequency: "",
        support_calls: "",
        payment_delay: "",
        subscription_type: "Basic",
        contract_length: "Monthly",
        total_spend: "",
        last_interaction: ""
    });

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const [shapExplanation, setShapExplanation] = useState([]);

    const handleChange = (event) => {

        const { name, value } = event.target;

        setFormData({
            ...formData,
            [name]: value
        });

    };


    const handleSubmit = async (event) => {

        event.preventDefault();

        setLoading(true);
        setError("");
        setResult(null);

        try {

            const response = await API.post(
                "predict/",
                {
                    age: Number(formData.age),
                    gender: formData.gender,
                    tenure: Number(formData.tenure),
                    usage_frequency: Number(formData.usage_frequency),
                    support_calls: Number(formData.support_calls),
                    payment_delay: Number(formData.payment_delay),
                    subscription_type: formData.subscription_type,
                    contract_length: formData.contract_length,
                    total_spend: Number(formData.total_spend),
                    last_interaction: Number(formData.last_interaction)
                }
            );

            setResult(response.data);

            setShapExplanation(response.data.shap_explanation || []);

        } catch (err) {

            if (err.response?.data?.errors) {
                setError(
                    JSON.stringify(err.response.data.errors)
                );
            } else {
                setError(
                    "Unable to connect to prediction API."
                );
            }

        } finally {

            setLoading(false);

        }
    };


    return (

        <div>

            <div className="page-header">

                <h1>Churn Prediction</h1>

                <p>
                    Enter customer information to predict churn risk.
                </p>

            </div>


            <div className="dashboard-card">

                <h2>Customer Information</h2>


                <form
                    onSubmit={handleSubmit}
                    className="prediction-form"
                >

                    <div className="form-grid">

                        <div className="form-group">
                            <label>Age</label>

                            <input
                                type="number"
                                name="age"
                                value={formData.age}
                                onChange={handleChange}
                                required
                            />
                        </div>


                        <div className="form-group">
                            <label>Gender</label>

                            <select
                                name="gender"
                                value={formData.gender}
                                onChange={handleChange}
                            >
                                <option value="Male">
                                    Male
                                </option>

                                <option value="Female">
                                    Female
                                </option>

                            </select>
                        </div>


                        <div className="form-group">
                            <label>Tenure</label>

                            <input
                                type="number"
                                name="tenure"
                                value={formData.tenure}
                                onChange={handleChange}
                                required
                            />
                        </div>


                        <div className="form-group">
                            <label>Usage Frequency</label>

                            <input
                                type="number"
                                name="usage_frequency"
                                value={formData.usage_frequency}
                                onChange={handleChange}
                                required
                            />
                        </div>


                        <div className="form-group">
                            <label>Support Calls</label>

                            <input
                                type="number"
                                name="support_calls"
                                value={formData.support_calls}
                                onChange={handleChange}
                                required
                            />
                        </div>


                        <div className="form-group">
                            <label>Payment Delay</label>

                            <input
                                type="number"
                                name="payment_delay"
                                value={formData.payment_delay}
                                onChange={handleChange}
                                required
                            />
                        </div>


                        <div className="form-group">
                            <label>Subscription Type</label>

                            <select
                                name="subscription_type"
                                value={formData.subscription_type}
                                onChange={handleChange}
                            >
                                <option value="Basic">
                                    Basic
                                </option>

                                <option value="Standard">
                                    Standard
                                </option>

                                <option value="Premium">
                                    Premium
                                </option>

                            </select>
                        </div>


                        <div className="form-group">
                            <label>Contract Length</label>

                            <select
                                name="contract_length"
                                value={formData.contract_length}
                                onChange={handleChange}
                            >
                                <option value="Monthly">
                                    Monthly
                                </option>

                                <option value="Quarterly">
                                    Quarterly
                                </option>

                                <option value="Annual">
                                    Annual
                                </option>

                            </select>
                        </div>


                        <div className="form-group">
                            <label>Total Spend</label>

                            <input
                                type="number"
                                step="0.01"
                                name="total_spend"
                                value={formData.total_spend}
                                onChange={handleChange}
                                required
                            />
                        </div>


                        <div className="form-group">
                            <label>Last Interaction</label>

                            <input
                                type="number"
                                name="last_interaction"
                                value={formData.last_interaction}
                                onChange={handleChange}
                                required
                            />
                        </div>

                    </div>


                    <button
                        type="submit"
                        className="predict-button"
                        disabled={loading}
                    >
                        {loading
                            ? "Predicting..."
                            : "Predict Churn"
                        }
                    </button>

                </form>


                {error && (

                    <div className="error-box">
                        {error}
                    </div>

                )}


                {result && (

                    <div className="prediction-result">

                        <h2>Prediction Result</h2>

                        <div className="result-label">

                            {result.prediction_label}

                        </div>

                        <p>

                            Churn Probability:

                            <strong>
                                {" "}
                                {(result.churn_probability * 100).toFixed(2)}%
                            </strong>

                        </p>

                    </div>

                )}

                {shapExplanation.length > 0 && (

                    <div className="shap-section">
                        <h3>Why did the model make this prediction?</h3>

                        {shapExplanation.map((item, index) => (
                            <div className="shap-item" key={index}>
                                <div className="shap-feature">
                                    {item.feature}
                                </div>

                                <div
                                    className={
                                        item.impact > 0
                                            ? "shap-impact positive"
                                            : "shap-impact negative"
                                    }
                                >
                                    {item.impact > 0 ? "+" : ""}
                                    {item.impact.toFixed(4)}
                                </div>
                            </div>
                        ))}

                        <div className="shap-chart-container">

                            <ResponsiveContainer
                                width="100%"
                                height={400}
                            >

                                <BarChart
                                    data={shapExplanation}
                                    layout="vertical"
                                    margin={{
                                        top: 10,
                                        right: 30,
                                        left: 30,
                                        bottom: 10
                                    }}
                                >

                                    <CartesianGrid
                                        strokeDasharray="3 3"
                                    />

                                    <XAxis
                                        type="number"
                                    />

                                    <YAxis
                                        type="category"
                                        dataKey="feature"
                                        width={260}
                                    />

                                    <Tooltip
                                        formatter={(value) =>
                                            Number(value).toFixed(4)
                                        }
                                    />

                                    <Bar
                                        dataKey="impact"
                                        name="SHAP Impact"
                                        radius={[0, 6, 6, 0]}
                                    />

                                </BarChart>

                            </ResponsiveContainer>

                        </div>
                    </div>
                )}

            </div>

        </div>

    );
}

export default Prediction;