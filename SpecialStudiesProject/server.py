from flask import Flask, render_template, request, jsonify
import os
import json
import importlib.util

# Import app.py dynamically from scripts folder
script_path = os.path.join(os.getcwd(), "scripts", "app.py")
spec = importlib.util.spec_from_file_location("app_logic", script_path)
app_logic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_logic)

app = Flask(__name__)

SCENARIOS_FOLDER = "scenarios"

# Ensure the folder exists
if not os.path.exists(SCENARIOS_FOLDER):
    os.makedirs(SCENARIOS_FOLDER)

# Load valid JSON scenario files
scenario_files = sorted(
    [f for f in os.listdir(SCENARIOS_FOLDER) if f.endswith(".json") and not f.startswith(".")]
)

scenarios = []
for file in scenario_files:
    file_path = os.path.join(SCENARIOS_FOLDER, file)
    if os.path.isfile(file_path):  # Ensure it's a file, not a folder
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "scenario" in data:
                    scenarios.append(data["scenario"])
                else:
                    print(f"Skipping {file}: Missing 'scenario' key")
        except (json.JSONDecodeError, PermissionError) as e:
            print(f"Skipping {file}: {e}")

user_choices = []
user_reasons = []
current_index = 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scenario")
def get_scenario():
    """Fetches the next scenario for the user."""
    global current_index
    if current_index < len(scenarios):
        return jsonify({"scenario": scenarios[current_index]})
    else:
        return jsonify({"scenario": "No more scenarios."})

@app.route("/submit", methods=["POST"])
def submit_choice():
    """Handles user choices and AI predictions while updating the frontend."""
    global current_index
    choice = request.form.get("choice")
    reason = request.form.get("reason")

    if choice and reason:
        user_choices.append(choice)
        user_reasons.append(reason)
        current_index += 1

    if current_index < len(scenarios) - 1:
        return jsonify({"next_scenario": scenarios[current_index]})
    else:
        # Use the AI model for prediction
        vectorizer, classifier = app_logic.train_classifier(scenarios[:-1], user_choices, user_reasons)
        final_scenario = scenarios[-1]
        predicted_choice = app_logic.explain_prediction(vectorizer, classifier, final_scenario)
        predicted_reason = app_logic.generate_reasoning(final_scenario, predicted_choice, user_reasons)

        return jsonify({
            "final_scenario": final_scenario,
            "prediction": predicted_choice,
            "reason": predicted_reason
        })

if __name__ == "__main__":
    app.run(debug=True)
