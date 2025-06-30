from flask import Flask, render_template, request
import os, uuid
from model.utils import predict_confidence_dcm, calculate_questionnaire_score, get_stage

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

questions = [
    "Do you experience hearing loss in one ear?",
    "Do you hear ringing or buzzing sounds (tinnitus) in one ear?",
    "Do you feel a sensation of fullness or blockage in one ear?",
    "Do you often feel unsteady or lose balance while walking?",
    "Have you experienced vertigo (spinning sensation)?",
    "Do loud sounds in one ear feel more irritating than usual?",
    "Do you have any facial numbness or weakness?",
    "Have you experienced nausea or vomiting with balance problems?",
    "Do you frequently have headaches without a clear cause?",
    "Have you noticed changes in your vision or blurred vision?"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", questions=questions)
def get_suggestions(stage):
    suggestions = {
        "Stage I - Early": [
            "🧑‍⚕️ Regular checkups every 6 months are advised.",
            "📊 Consider tracking hearing loss progression.",
            "💊 No treatment necessary unless symptoms worsen."
        ],
        "Stage II - Moderate": [
            "🧠 Schedule MRI every 3–6 months to monitor growth.",
            "🎯 ENT and Neurology consultation recommended.",
            "💊 Medication for dizziness or tinnitus may help."
        ],
        "Stage III - Advanced": [
            "🏥 Surgery or radiation therapy evaluation needed.",
            "🩺 Multidisciplinary team review essential.",
            "🛏️ Consider lifestyle support and rehabilitation planning."
        ],
        "Stage IV - Severe": [
            "⚠️ Immediate treatment required (likely surgery or radiosurgery).",
            "👨‍⚕️ Urgent hospital referral and intervention.",
            "🧑‍⚕️ Long-term care support should be arranged."
        ]
    }
    return suggestions.get(stage, ["Avoid long exposure to loud noise or headphones.",
                                   "Incorporate vestibular balance exercises such as Tai Chi or Yoga.",
                                   "If balance issues or ringing in the ear persist, an ENT or neurologist consult is advised — even if the MRI is clear."])

@app.route("/result", methods=["POST"])
def result():
    # Save uploaded DICOM file
    uploaded_file = request.files["dcm_file"]
    dcm_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.dcm")
    uploaded_file.save(dcm_path)

    # Predict MRI confidence score
    mri_score = predict_confidence_dcm(dcm_path)

    # Questionnaire score
    responses = {f"q{i+1}": request.form[f"q{i+1}"] for i in range(10)}
    q_score = calculate_questionnaire_score(responses)

    # Final combined score
    final_score = round((0.9 * mri_score) + (0.1 * q_score), 2)
    stage = get_stage(final_score)
    suggestions = get_suggestions(stage)
    return render_template("result.html",
                           mri_score=mri_score,
                           q_score=q_score,
                           final_score=final_score,
                           stage=stage,
                           suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)
