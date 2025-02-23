from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import gdown
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Dictionary of public Google Drive links for reference signatures
drive_links = {
    1: "https://drive.google.com/file/d/1xtzL6-TpN4EVyaFUF4MM4ssjAZqZutF8/view?usp=drive_link",
    2: "https://drive.google.com/file/d/1UpPfOlDXoWwB5Ub530uhUrOVnUnWYvpQ/view?usp=drive_link",
    3: "https://drive.google.com/file/d/1-M_PND4PK3tSLY705olnsswOk5bNoOFa/view?usp=drive_link",
    4: "https://drive.google.com/file/d/1FL1uLEXlWW-nQYNoaBARiVs0N0XAwsvW/view?usp=drive_link",
    5: "https://drive.google.com/file/d/1nZhl1CkvuH-KA4ErAslD-91W2QnBajhx/view?usp=drive_link",
    6: "https://drive.google.com/file/d/1SHEgykTZN9lGdDaR6PTl9P01-Zlpu6cZ/view?usp=drive_link",
    7: "https://drive.google.com/file/d/1gRE9SmvT7OBw8JYCyx7ehMs3lBpiX-Bp/view?usp=drive_link"
}

# ✅ Function to extract file ID from Google Drive link
def extract_file_id(drive_url):
    return drive_url.split("/d/")[1].split("/view")[0]

# ✅ Function to download a file from Google Drive
def download_from_drive(file_id, save_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", save_path, quiet=False)
    return save_path

# ✅ ORB Feature Matching for Signature Comparison
def orb_similarity(img1, img2, distance_threshold=50):
    gray1, gray2 = [
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        for img in [img1, img2]
    ]

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    good_matches = [m for m in matches if m.distance < distance_threshold]
    similarity = len(good_matches) / len(matches) if matches else 0
    return similarity

# ✅ API Route to Verify Signature
@app.route("/verify", methods=["POST"])
def verify_signature():
    if "image" not in request.files or "reference_number" not in request.form:
        return jsonify({"error": "Missing image or reference number"}), 400

    image_file = request.files["image"]
    reference_number = int(request.form["reference_number"])

    if reference_number not in drive_links:
        return jsonify({"error": "Invalid reference number"}), 400

    # Save the uploaded image
    document_image_path = "uploaded_document.jpg"
    image_file.save(document_image_path)

    # Download reference signature
    file_id = extract_file_id(drive_links[reference_number])
    reference_image_path = f"reference_signature_{reference_number}.jpg"
    download_from_drive(file_id, reference_image_path)

    # Load images
    document_img = cv2.imread(document_image_path)
    reference_img = cv2.imread(reference_image_path)

    if document_img is None or reference_img is None:
        return jsonify({"error": "Error loading images"}), 400

    # Compute similarity
    similarity = orb_similarity(document_img, reference_img)
    similarity_percentage = round(similarity * 100, 2)

    # Classification based on similarity score
    if similarity_percentage > 55:
        classification = "Matched"
    elif 40 <= similarity_percentage <= 55:
        classification = "Manual Check Recommended"
    else:
        classification = "Not Matched"

    # Return the JSON response
    return jsonify({
        "similarity_score": similarity_percentage,
        "classification": classification
    })

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
