from flask import Flask, request, jsonify
#Flask: A Python framework used to build web applications and APIs.
#request: Allows us to handle incoming data (e.g., from JotForm).
#jsonify: Converts Python data into JSON format so it can be sent back to JotForm.

import pandas as pd
import numpy as np

#app = Flask(__name__)


# Load modules from CSV
modules_df = pd.read_csv("modules.csv")
module_vectors = modules_df[["informed", "prepared", "hopeful", "support", "tools"]].values
print(module_vectors)

"""
# Function to calculate recommendation scores
def recommend_modules(user_data):
    # Extract user recovery goals and special interests
    goals = {goal: int(user_data.get(goal, 0)) for goal in ["informed", "prepared", "hopeful", "support", "tools"]}
    special_interests = user_data.get("special_interests", "").split(",")

    max_value = max(goals.values())  # Find the highest goal score
    top_goals = [goal for goal, value in goals.items() if value == max_value]  # Find all top goals

    
    # Score each module (weighting top goal more)
    modules_df["score"] = (
        0.6 * modules_df[top_goal] +  # Higher weight for top goal
        0.4 * modules_df[["informed", "prepared", "hopeful", "support", "tools"]].sum(axis=1)
    )

    # Sort modules by best score (higher is better)
    top_modules = modules_df.nlargest(5, "score")[["module_name"]]

    # Add special interest modules
    special_modules = [mod for mod in special_interests if mod in modules_df["module_name"].values]

    return {"recommendations": top_modules["module_name"].tolist() + special_modules}
"""

test_user_2 = {
    "informed": 1,
    "prepared": 2,
    "hopeful": 3,
    "support": 1,
    "tools": 3,
    "special_interests": "BIPOC,transgender"
}

test_user_1 = {
    "informed": 3,
    "prepared": 2,
    "hopeful": 1,
    "support": 1,
    "tools": 3,
    "special_interests": "BIPOC,transgender"
}

user_vector = np.array([int(test_user_1.get(goal, 0)) for goal in ["informed", "prepared", "hopeful", "support", "tools"]])
print(user_vector)
distances = np.linalg.norm(module_vectors - user_vector, axis=1)
print(distances)
modules_df["distance"] = distances
print(modules_df)
top_modules = modules_df.nsmallest(5, "distance")
print(top_modules)


def recommend_modules(user_data):
    # Extract user recovery goals as a NumPy array
    user_vector = np.array([int(user_data.get(goal, 0)) for goal in ["informed", "prepared", "hopeful", "support", "tools"]])

    # Convert module scores to a NumPy array
    module_vectors = modules_df[["informed", "prepared", "hopeful", "support", "tools"]].values  

    # Compute Euclidean distance for each module
    distances = np.linalg.norm(module_vectors - user_vector, axis=1)

    # Add distances to DataFrame
    modules_df["distance"] = distances

    # Sort modules by **smallest** Euclidean distance (most similar)
    top_modules = modules_df.nsmallest(5, "distance")[["module_name"]]

    # Special interest modules (direct match)
    special_interests = user_data.get("special_interests", "").split(",")
    special_modules = [mod for mod in special_interests if mod in modules_df["module_name"].values]

    return {"recommendations": top_modules["module_name"].tolist() + special_modules}

print(recommend_modules(test_user_2))

#@app.route("/webhook", methods=["POST"])
def handle_webhook():
    user_data = request.json  # Receive JotForm data
    recommendations = recommend_modules(user_data)
    return jsonify(recommendations)  # Send response back to JotForm

#if __name__ == "__main__":
    app.run(port=5000, debug=True)
