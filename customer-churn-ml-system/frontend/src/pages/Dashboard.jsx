import { useEffect, useState } from "react";
import {
    PieChart,
    Pie,
    Cell,
    Tooltip,
    Legend,
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid
} from "recharts";

import API from "../services/api";
import StatCard from "../components/StatCard";

function Dashboard() {

    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    const [featureImportance, setFeatureImportance] = useState([]);

    useEffect(() => {

        const fetchDashboardData = async () => {

            try {

                const response = await API.get(
                    "predict/history/"
                );

                setPredictions(
                    response.data.predictions
                );

                const featureResponse = await API.get(
                    "predict/feature-importance/"
                );

                setFeatureImportance(
                    featureResponse.data
                );

            } catch (err) {

                console.error(err);

                setError(
                    "Unable to load dashboard data."
                );

            } finally {

                setLoading(false);

            }
        };

        fetchDashboardData();

    }, []);


    if (loading) {

        return (
            <div>

                <div className="page-header">
                    <h1>Dashboard</h1>
                    <p>Loading analytics...</p>
                </div>

            </div>
        );

    }


    const totalPredictions = predictions.length;

    const churnedCustomers = predictions.filter(
        (item) => item.prediction === 1
    ).length;

    const safeCustomers = predictions.filter(
        (item) => item.prediction === 0
    ).length;

    const churnRate = totalPredictions > 0
        ? ((churnedCustomers / totalPredictions) * 100).toFixed(2)
        : "0.00";


    const chartData = [
        {
            name: "Churn",
            value: churnedCustomers
        },
        {
            name: "Not Churn",
            value: safeCustomers
        }
    ];


    return (

        <div>

            <div className="page-header">

                <h1>Dashboard</h1>

                <p>
                    Monitor customer churn predictions
                    and risk statistics.
                </p>

            </div>


            {error && (

                <div className="error-box">
                    {error}
                </div>

            )}


            <div className="stats-grid">

                <StatCard
                    title="Total Predictions"
                    value={totalPredictions}
                    icon="👥"
                />

                <StatCard
                    title="Churned Customers"
                    value={churnedCustomers}
                    icon="⚠️"
                />

                <StatCard
                    title="Safe Customers"
                    value={safeCustomers}
                    icon="✅"
                />

                <StatCard
                    title="Churn Rate"
                    value={`${churnRate}%`}
                    icon="📈"
                />

            </div>


            <div className="dashboard-card chart-card">

                <h2>Churn Distribution</h2>

                <p>
                    Distribution of predictions made by the ML model.
                </p>


                <div className="chart-container">

                    {totalPredictions > 0 ? (

                        <ResponsiveContainer
                            width="100%"
                            height={350}
                        >

                            <PieChart>

                                <Pie
                                    data={chartData}
                                    dataKey="value"
                                    nameKey="name"
                                    cx="50%"
                                    cy="50%"
                                    outerRadius={120}
                                    label
                                >

                                    {chartData.map(
                                        (entry, index) => (
                                            <Cell
                                                key={`cell-${index}`}
                                            />
                                        )
                                    )}

                                </Pie>

                                <Tooltip />

                                <Legend />

                            </PieChart>

                        </ResponsiveContainer>

                    ) : (

                        <p className="empty-message">
                            No prediction data available yet.
                        </p>

                    )}

                </div>

            </div>

            <div className="dashboard-card chart-card">

                <h2>Global SHAP Feature Importance</h2>

                <p>
                    Average absolute SHAP impact of features on model predictions.
                </p>

                <div className="chart-container">

                    {featureImportance.length > 0 ? (

                        <ResponsiveContainer
                            width="100%"
                            height={450}
                        >

                            <BarChart
                                data={featureImportance}
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
                                    width={240}
                                />

                                <Tooltip
                                    formatter={(value) =>
                                        Number(value).toFixed(4)
                                    }
                                />

                                <Bar
                                    dataKey="importance"
                                    name="Mean |SHAP|"
                                    radius={[0, 6, 6, 0]}
                                />

                            </BarChart>

                        </ResponsiveContainer>

                    ) : (

                        <p className="empty-message">
                            No feature importance data available.
                        </p>

                    )}

                </div>

            </div>

        </div>

    );
}

export default Dashboard;