# app.py
import os
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import warnings
import numpy as np
import pdf_parser
from werkzeug.utils import secure_filename

# Compatibility fix for numpy >= 1.20 with older libraries (LIME, etc.)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'complex'):
        np.complex = complex
    if not hasattr(np, 'object'):
        np.object = object
    if not hasattr(np, 'str'):
        np.str = str
    if not hasattr(np, 'long'):
        np.long = int
    if not hasattr(np, 'unicode'):
        np.unicode = str

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import joblib
import pandas as pd
from utils import (EXPECTED_FEATURES, validate_and_prepare_df,
                   single_input_to_df, generate_care_plan,
                   generate_shap_plot, generate_lime_plot,
                   analyze_dataset_features, generate_feature_importance_plot,
                   generate_distribution_plot, calculate_dataset_statistics,
                   save_predictions_csv, generate_dice_bcf, generate_anchor_rule)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "change_this_randomly_for_prod"  # change for production

DB_PATH = "users.db"
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
TRAIN_CSV_PATH = os.path.join(MODEL_DIR, "train_data_sample.csv")

# Create DB if missing
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

# Ensure model files exist
if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model or scaler not found. Run train_model.py first to generate models/ files.")

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

# DB helpers
def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    return row

def create_user(email, password_plain):
    pw_hash = generate_password_hash(password_plain)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, pw_hash))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, str(e)
    finally:
        conn.close()

# Load training raw df for LIME (validate)
if not os.path.exists(TRAIN_CSV_PATH):
    raise FileNotFoundError("Training CSV not found at data/pima_diabetes.csv; required for LIME explainer.")
train_df_raw = pd.read_csv(TRAIN_CSV_PATH)
train_df_raw = train_df_raw.rename(columns=lambda c: c.strip())
# Validate training df columns
missing = [c for c in EXPECTED_FEATURES + ["Diabetes_binary"] if c not in train_df_raw.columns]
if missing:
    raise ValueError(f"Training CSV missing expected columns: {missing}")
# drop rows with NaNs in expected features for LIME creation
train_df_raw = train_df_raw.dropna(subset=EXPECTED_FEATURES)

# --- Routes ---
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if not email or not password:
            flash("Provide email and password", "danger")
            return render_template("register.html")
        ok, err = create_user(email, password)
        if not ok:
            flash("Unable to create user: " + (err or "email maybe taken"), "danger")
            return render_template("register.html")
        flash("Registration successful. You can login now.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        user = get_user_by_email(email)
        if user:
            uid, uemail, pw_hash = user
            if check_password_hash(pw_hash, password):
                session["user_id"] = uid
                session["user_email"] = uemail
                return redirect(url_for("dashboard"))
        flash("Invalid email or password", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session.get("user_email"))

@app.route('/comparisons')
def comparisons():
    if "user_id" not in session:
        return redirect(url_for("login"))
    # Load combined CSV if available
    csv_path = "literature_comparison/comparison/combined_comparison.csv"
    if not os.path.exists(csv_path):
        flash('Combined comparison CSV not found. Please run the comparison scripts first.', 'warning')
        return redirect(url_for('dashboard'))
    df = pd.read_csv(csv_path)
    # Convert to HTML table (Bootstrap)
    table_html = df.to_html(classes='table table-striped table-sm', index=False, justify='left')
    return render_template('comparisons.html', table_html=table_html)

@app.route('/comparison_image/<path:filename>')
def comparison_image(filename):
    # Serve image files from literature_comparison/comparison
    return send_from_directory(os.path.join(os.getcwd(), 'literature_comparison', 'comparison'), filename)


@app.route("/upload", methods=["GET","POST"])
def upload():
    if "user_id" not in session:
        return redirect(url_for("login"))
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("Please upload a CSV file", "warning")
            return redirect(request.url)
        try:
            # Read and validate CSV
            df = pd.read_csv(file)
            df_clean = validate_and_prepare_df(df)
            
            # Make predictions
            X = df_clean.values
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]  # Probability of diabetic class
            
            # Basic statistics
            count_diabetic = int(np.sum(preds == 1))
            total = len(preds)
            percent = round(count_diabetic/total * 100, 2)
            
            # Feature importance analysis
            feature_importance = analyze_dataset_features(model, scaler, df_clean, preds, probs)
            
            # Generate visualizations
            feature_plot = None
            if feature_importance:
                feature_plot = generate_feature_importance_plot(feature_importance)
            
            distribution_plot = generate_distribution_plot(preds, probs)
            
            # Calculate dataset statistics
            statistics = calculate_dataset_statistics(df_clean)
            
            # Save predictions to CSV
            csv_filename = save_predictions_csv(df, preds, probs)
            
            result = {
                "total": total,
                "predicted_diabetic": count_diabetic,
                "percentage": percent,
                "feature_importance": feature_importance,
                "feature_plot": feature_plot,
                "distribution_plot": distribution_plot,
                "statistics": statistics,
                "csv_filename": csv_filename,
                "csv_path": True if csv_filename else False
            }
            
            flash(f"Successfully analyzed {total} records!", "success")
            
        except Exception as e:
            flash("Error processing CSV: " + str(e), "danger")
            import traceback
            print(traceback.format_exc())
    return render_template("upload.html", result=result)

@app.route("/self", methods=["GET","POST"])
def self_monitor():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    prediction = None
    care_plan = None
    shap_img = None
    lime_img = None
    dice_exp = None
    anchor_rule = None
    form_data = {} # To keep input values
    
    if request.method == "POST":
        form_data = request.form.to_dict() # Persistence
        df = single_input_to_df(form_data)
        try:
            df_valid = validate_and_prepare_df(df)
            X_raw = df_valid  # DataFrame single-row in raw units
            X_scaled = scaler.transform(X_raw.values)
            prob = float(model.predict_proba(X_scaled)[0][1])
            pred_label = int(model.predict(X_scaled)[0])
            prob_pct = round(prob * 100, 2)
            prediction = {"probability": prob_pct, "label": "Diabetic" if pred_label == 1 else "Not Diabetic"}
            
            # Care Plan (Critical)
            try:
                import shap
                ex = shap.TreeExplainer(model)
                shap_vals = ex.shap_values(X_scaled)
                care_plan = generate_care_plan(df_valid.iloc[0].to_dict(), shap_vals, prob_pct)
            except Exception as e:
                print(f"Error generating care plan: {e}")
                # If care plan fails, we can't show full results, so let it bubble to main except OR provide dummy
                raise e

            # Auxiliary XAI (Non-Critical) - failures here shouldn't stop the prediction showing
            try:
                shap_rel = generate_shap_plot(model, scaler, X_raw, feature_names=EXPECTED_FEATURES)
                shap_img = "/" + shap_rel
            except Exception as e:
                print(f"SHAP Plot Error: {e}")

            try:
                lime_rel = generate_lime_plot(model, scaler, X_raw, train_df_raw, feature_names=EXPECTED_FEATURES)
                lime_img = "/" + lime_rel
            except Exception as e:
                 print(f"LIME Error: {e}")

            try:
                dice_exp = generate_dice_bcf(model, X_raw, train_df_raw)
            except Exception as e:
                print(f"DiCE Error: {e}")

            try:
                anchor_rule = generate_anchor_rule(model, X_raw, train_df_raw.values, feature_names=EXPECTED_FEATURES)
            except Exception as e:
                print(f"Anchors Error: {e}")

        except Exception as e:
            flash("Error during prediction: " + str(e), "danger")
            prediction = None # Prevent template from trying to render partial state
            import traceback
            traceback.print_exc()
            
    return render_template("self_monitor.html",
                           prediction=prediction,
                           care_plan=care_plan,
                           shap_img=shap_img,
                           lime_img=lime_img,
                           dice_exp=dice_exp,
                           anchor_rule=anchor_rule,
                           form_data=form_data)

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    file = request.files.get("pdf_file")
    if not file or not file.filename.endswith(".pdf"):
        flash("Please upload a valid PDF file", "danger")
        return redirect(url_for("self_monitor"))
    
    # Save temp
    temp_path = os.path.join("static", secure_filename(file.filename))
    file.save(temp_path)
    
    try:
        extracted_data = pdf_parser.extract_data_from_pdf(temp_path)
        # Pre-fill form_data for the template.
        # We render self_monitor with this data
        if not extracted_data:
            flash("Could not extract data from PDF. Please enter manually.", "warning")
        else:
            flash("Data extracted successfully! Please verify fields.", "success")
        
        return render_template("self_monitor.html", form_data=extracted_data)
        
    except Exception as e:
        flash("Error processing PDF: " + str(e), "danger")
        return redirect(url_for("self_monitor"))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/download_sample_csv")
def download_sample_csv():
    """Generate and download a sample CSV template"""
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    # Create sample CSV
    # Based on new columns (21 features)
    sample_data = {feat: [0]*3 for feat in EXPECTED_FEATURES}
    # Customize a few for realism if needed, but 0s are fine for template structure
    sample_df = pd.DataFrame(sample_data)
    
    # Save to temp location
    temp_path = os.path.join("static", "dataset_analysis", "sample_template.csv")
    sample_df.to_csv(temp_path, index=False)
    
    return send_from_directory(
        os.path.join(os.getcwd(), 'static', 'dataset_analysis'),
        'sample_template.csv',
        as_attachment=True,
        download_name='diabetes_dataset_template.csv'
    )

@app.route("/download_results/<filename>")
def download_results(filename):
    """Download the predictions CSV file"""
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    return send_from_directory(
        os.path.join(os.getcwd(), 'static', 'dataset_analysis'),
        filename,
        as_attachment=True
    )

# static route for saved images (Flask serves /static automatically)
if __name__ == "__main__":
    app.run(debug=True)
