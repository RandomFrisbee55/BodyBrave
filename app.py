from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load modules from CSV
modules_df = pd.read_csv("modules.csv")

# Function to calculate recommendation scores
def recommend_modules(user_data):
    # Extract user recovery goals and special interests
    goals = {goal: int(user_data.get(goal, 0)) for goal in ["informed", "prepared", "hopeful", "support", "tools"]}
    special_interests = user_data.get("special_interests", "").split(",")

    # Find the highest-rated goals (all top values)
    max_goal_value = max(goals.values())  # Get the highest rating given
    top_goals = [goal for goal, value in goals.items() if value == max_goal_value]  # Get all goals with highest rating

    # Score each module (weighting top goals more)
    goal_weights = np.array([0.6 if col in top_goals else 0.4 for col in ["informed", "prepared", "hopeful", "support", "tools"]])
    module_vectors = modules_df[["informed", "prepared", "hopeful", "support", "tools"]].values
    user_vector = np.array([goals[col] for col in ["informed", "prepared", "hopeful", "support", "tools"]])

    # Compute Euclidean distance
    distances = np.linalg.norm(module_vectors - user_vector, axis=1)

    # Get top 5 closest modules
    top_modules = modules_df.iloc[np.argsort(distances)[:5]][["module_name"]]

    # Add special interest modules
    special_modules = [mod for mod in special_interests if mod in modules_df["module_name"].values]

    return {"recommendations": top_modules["module_name"].tolist() + special_modules}

# Webhook route
@app.route("/webhook", methods=["POST"])
def handle_webhook():
    user_data = request.json  # Receive JotForm data
    recommendations = recommend_modules(user_data)  # Get module recommendations
    return jsonify(recommendations)  # Send response back to JotForm

if __name__ == "__main__":
    app.run(port=5000, debug=True)
