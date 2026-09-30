from flask import Flask, render_template, request

import pandas as pd
import joblib
import os


app = Flask(__name__)


# ==================================================
# Load trained model and preprocessor
# ==================================================

MODEL_PATH = os.path.join("models", "model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# ==================================================
# Home Page
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# Team Page
# ==================================================

@app.route("/team")
def team():
    return render_template("team.html")


# ==================================================
# Details Page
# ==================================================

@app.route("/details")
def details():
    return render_template("details.html")


# ==================================================
# Prediction Page
# ==================================================

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


# ==================================================
# Prediction Processing
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # --------------------------------------------------
        # Read preset selection
        # --------------------------------------------------

        preset = request.form.get(
            "preset",
            ""
        ).strip().lower()


        # --------------------------------------------------
        # Required visible fields
        # --------------------------------------------------

        required_fields = [
            "ProductRelated",
            "ProductRelated_Duration",
            "Informational",
            "Administrative",
            "PageValues",
            "BounceRates",
            "ExitRates",
            "Weekend",
            "Month",
            "VisitorType"
        ]


        # --------------------------------------------------
        # Check required fields
        # --------------------------------------------------

        missing_fields = []

        for field in required_fields:

            value = request.form.get(
                field,
                ""
            ).strip()

            if value == "":
                missing_fields.append(field)


        # --------------------------------------------------
        # If fields are missing, return error
        # --------------------------------------------------

        if missing_fields:

            return render_template(

                "prediction.html",

                prediction="Incomplete Input",

                result_message=(
                    "Please enter or select all required "
                    "shopping-session details before "
                    "generating a prediction."
                ),

                probability=None,

                result_icon="!",

                error=True

            )


        # ==================================================
        # Read visible input values
        # ==================================================

        ProductRelated = float(
            request.form.get("ProductRelated")
        )

        ProductRelated_Duration = float(
            request.form.get(
                "ProductRelated_Duration"
            )
        )

        Informational = float(
            request.form.get("Informational")
        )

        Administrative = float(
            request.form.get("Administrative")
        )

        PageValues = float(
            request.form.get("PageValues")
        )

        BounceRates = float(
            request.form.get("BounceRates")
        )

        ExitRates = float(
            request.form.get("ExitRates")
        )

        Month = request.form.get(
            "Month"
        )

        VisitorType = request.form.get(
            "VisitorType"
        )

        Weekend_value = request.form.get(
            "Weekend"
        )


        # ==================================================
        # Convert Weekend to Boolean
        # ==================================================

        Weekend = Weekend_value.lower() in [
            "true",
            "1",
            "yes",
            "on"
        ]


        # ==================================================
        # Hidden model features
        #
        # These features are part of the trained model
        # but are kept at representative default values
        # for the simplified web interface.
        # ==================================================

        Administrative_Duration = 75

        Informational_Duration = 30

        SpecialDay = 0

        OperatingSystems = 2

        Browser = 2

        Region = 3

        TrafficType = 4


        # ==================================================
        # Create input DataFrame
        # ==================================================

        input_data = pd.DataFrame({

            "Administrative": [
                Administrative
            ],

            "Administrative_Duration": [
                Administrative_Duration
            ],

            "Informational": [
                Informational
            ],

            "Informational_Duration": [
                Informational_Duration
            ],

            "ProductRelated": [
                ProductRelated
            ],

            "ProductRelated_Duration": [
                ProductRelated_Duration
            ],

            "BounceRates": [
                BounceRates
            ],

            "ExitRates": [
                ExitRates
            ],

            "PageValues": [
                PageValues
            ],

            "SpecialDay": [
                SpecialDay
            ],

            "Month": [
                Month
            ],

            "OperatingSystems": [
                OperatingSystems
            ],

            "Browser": [
                Browser
            ],

            "Region": [
                Region
            ],

            "TrafficType": [
                TrafficType
            ],

            "VisitorType": [
                VisitorType
            ],

            "Weekend": [
                Weekend
            ]

        })


        # ==================================================
        # Preprocess input
        # ==================================================

        processed_data = preprocessor.transform(
            input_data
        )


        # ==================================================
        # Actual ML prediction
        # ==================================================

        prediction_value = model.predict(
            processed_data
        )[0]


        # ==================================================
        # Actual ML probability
        # ==================================================

        probabilities = model.predict_proba(
            processed_data
        )[0]

        purchase_probability = probabilities[1] * 100


        # ==================================================
        # IMPORTANT
        #
        # Preset buttons only provide convenient example
        # input values.
        #
        # The prediction and probability ALWAYS come
        # from the trained Decision Tree model.
        # ==================================================

        if prediction_value == 1:

            prediction_text = "Purchase Likely"

            result_message = (
                "The trained Decision Tree model predicts "
                "that this shopping session is likely to "
                "result in a purchase."
            )

            result_icon = "✓"

        else:

            prediction_text = "Purchase Unlikely"

            result_message = (
                "The trained Decision Tree model predicts "
                "that this shopping session is unlikely "
                "to result in a purchase."
            )

            result_icon = "×"


        # ==================================================
        # Return result
        # ==================================================

        return render_template(

            "prediction.html",

            prediction=prediction_text,

            result_message=result_message,

            probability=round(
                purchase_probability,
                2
            ),

            result_icon=result_icon,

            error=False

        )


    except ValueError:

        return render_template(

            "prediction.html",

            prediction="Invalid Input",

            result_message=(
                "Please enter valid numeric values "
                "in all numeric fields."
            ),

            probability=None,

            result_icon="!",

            error=True

        )


    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return render_template(

            "prediction.html",

            prediction="Prediction Error",

            result_message=(
                "An unexpected error occurred while "
                "processing the prediction."
            ),

            probability=None,

            result_icon="!",

            error=True

        )


# ==================================================
# Run Flask Application
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )