import { NavLink } from "react-router-dom";

function Sidebar() {

    return (
        <aside className="sidebar">

            <div className="logo">
                ChurnAI
            </div>

            <nav>

                <NavLink to="/">
                    📊 Dashboard
                </NavLink>

                <NavLink to="/prediction">
                    🔮 Prediction
                </NavLink>

                <NavLink to="/history">
                    📋 History
                </NavLink>

            </nav>

        </aside>
    );
}

export default Sidebar;