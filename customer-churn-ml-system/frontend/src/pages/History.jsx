import { useEffect, useState } from "react";
import API from "../services/api";

function History() {

    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    const fetchHistory = async () => {

        try {

            const response = await API.get("predict/history/");

            setPredictions(response.data.predictions);

        } catch (err) {

            console.error(err);

            setError(
                "Unable to load prediction history."
            );

        } finally {

            setLoading(false);

        }
    };


    useEffect(() => {

        fetchHistory();

    }, []);


    if (loading) {

        return (
            <div>

                <div className="page-header">
                    <h1>Prediction History</h1>
                    <p>Loading prediction records...</p>
                </div>

            </div>
        );

    }


    return (

        <div>

            <div className="page-header">

                <h1>Prediction History</h1>

                <p>
                    View previous customer churn predictions.
                </p>

            </div>


            {error && (

                <div className="error-box">
                    {error}
                </div>

            )}


            <div className="dashboard-card history-card">

                <div className="history-header">

                    <h2>Prediction Records</h2>

                    <span>
                        Total: {predictions.length}
                    </span>

                </div>


                {predictions.length === 0 ? (

                    <p className="empty-message">
                        No prediction records found.
                    </p>

                ) : (

                    <div className="table-container">

                        <table>

                            <thead>

                                <tr>

                                    <th>ID</th>
                                    <th>Age</th>
                                    <th>Gender</th>
                                    <th>Tenure</th>
                                    <th>Support Calls</th>
                                    <th>Payment Delay</th>
                                    <th>Contract</th>
                                    <th>Prediction</th>
                                    <th>Probability</th>
                                    <th>Created At</th>

                                </tr>

                            </thead>


                            <tbody>

                                {predictions.map((prediction) => (

                                    <tr key={prediction.id}>

                                        <td>
                                            {prediction.id}
                                        </td>

                                        <td>
                                            {prediction.age}
                                        </td>

                                        <td>
                                            {prediction.gender}
                                        </td>

                                        <td>
                                            {prediction.tenure}
                                        </td>

                                        <td>
                                            {prediction.support_calls}
                                        </td>

                                        <td>
                                            {prediction.payment_delay}
                                        </td>

                                        <td>
                                            {prediction.contract_length}
                                        </td>

                                        <td>

                                            <span
                                                className={
                                                    prediction.prediction === 1
                                                        ? "status-churn"
                                                        : "status-safe"
                                                }
                                            >

                                                {prediction.prediction_label}

                                            </span>

                                        </td>

                                        <td>

                                            {(
                                                prediction.churn_probability * 100
                                            ).toFixed(2)}%

                                        </td>

                                        <td>

                                            {new Date(
                                                prediction.created_at
                                            ).toLocaleString()}

                                        </td>

                                    </tr>

                                ))}

                            </tbody>

                        </table>

                    </div>

                )}

            </div>

        </div>

    );
}

export default History;