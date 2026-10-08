import { BrowserRouter, Routes, Route } from "react-router-dom";

import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";

import Dashboard from "./pages/Dashboard";
import Prediction from "./pages/Prediction";
import History from "./pages/History";

function App() {
    return (
        <BrowserRouter>

            <div className="app">

                <Sidebar />

                <div className="main-content">

                    <Navbar />

                    <main className="page-content">

                        <Routes>

                            <Route
                                path="/"
                                element={<Dashboard />}
                            />

                            <Route
                                path="/prediction"
                                element={<Prediction />}
                            />

                            <Route
                                path="/history"
                                element={<History />}
                            />

                        </Routes>

                    </main>

                </div>

            </div>

        </BrowserRouter>
    );
}

export default App;