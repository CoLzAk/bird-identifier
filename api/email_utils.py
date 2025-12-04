# api/email_utils.py
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import base64


# Load species data
def load_species_data() -> Dict[str, str]:
    """Load species data from species.json and return a mapping of scientific names to common names."""
    species_file = Path(__file__).parent / "species" / "species.json"

    if not species_file.exists():
        return {}

    try:
        with open(species_file, 'r', encoding='utf-8') as f:
            species_list = json.load(f)
            return {sp['name']: sp['common_name'] for sp in species_list}
    except Exception as e:
        print(f"⚠️ Failed to load species data: {e}")
        return {}


# Cache species data
SPECIES_COMMON_NAMES = load_species_data()


async def send_bird_detection_email(
    image_bytes: bytes,
    predictions: List[Dict[str, any]],
    filename: str = "bird_detection.jpg"
):
    """
    Send an email notification with bird detection results.

    Args:
        image_bytes: Image data as bytes
        predictions: List of predictions with 'species' and 'probability' keys
        filename: Name for the attached image file
    """
    # Get SMTP configuration from environment
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    email_from = os.getenv("EMAIL_FROM")
    email_recipient = os.getenv("EMAIL_RECIPIENT")

    # Validate configuration
    if not all([smtp_host, smtp_username, smtp_password, email_from, email_recipient]):
        print("⚠️ Email configuration incomplete, skipping email notification")
        return

    # Create message
    msg = MIMEMultipart('related')
    msg['Subject'] = "🐦 Oiseau détecté"
    msg['From'] = email_from
    msg['To'] = email_recipient

    # Create HTML body
    html_body = create_html_body(predictions)

    # Attach HTML
    html_part = MIMEText(html_body, 'html', 'utf-8')
    msg.attach(html_part)

    # Attach image
    image = MIMEImage(image_bytes)
    image.add_header('Content-ID', '<bird_image>')
    image.add_header('Content-Disposition', 'inline', filename=filename)
    msg.attach(image)

    try:
        # Send email
        await aiosmtplib.send(
            msg,
            hostname=smtp_host,
            port=smtp_port,
            username=smtp_username,
            password=smtp_password,
            start_tls=True
        )
        print(f"✅ Email sent successfully to {email_recipient}")
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")


def get_display_name(species_name: str) -> str:
    """Get the display name for a species (French common name if available, otherwise scientific name)."""
    if species_name in SPECIES_COMMON_NAMES:
        return SPECIES_COMMON_NAMES[species_name]
    # Fallback: format scientific name
    return species_name.replace('_', ' ').title()


def create_html_body(predictions: List[Dict[str, any]]) -> str:
    """
    Create HTML email body with bird detection results.

    Args:
        predictions: List of predictions with 'species' and 'probability' keys

    Returns:
        HTML string for email body
    """
    # Get top prediction
    top_prediction = predictions[0] if predictions else None
    other_predictions = predictions[1:3] if len(predictions) > 1 else []

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }}
            .image-container {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .bird-image {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }}
            .top-prediction {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
            }}
            .top-prediction-species {{
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .top-prediction-probability {{
                font-size: 24px;
                opacity: 0.9;
            }}
            .other-predictions {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
            }}
            .other-predictions h3 {{
                color: #6c757d;
                font-size: 16px;
                margin-top: 0;
                margin-bottom: 15px;
            }}
            .prediction-item {{
                display: flex;
                justify-content: space-between;
                padding: 10px;
                margin-bottom: 8px;
                background-color: white;
                border-radius: 5px;
                border-left: 3px solid #667eea;
            }}
            .prediction-species {{
                font-size: 16px;
                font-weight: 500;
                color: #2c3e50;
            }}
            .prediction-probability {{
                font-size: 16px;
                color: #667eea;
                font-weight: 600;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                color: #6c757d;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐦 Oiseau détecté</h1>

            <div class="image-container">
                <img src="cid:bird_image" alt="Oiseau détecté" class="bird-image">
            </div>
    """

    if top_prediction:
        species_name = get_display_name(top_prediction['species'])
        probability = top_prediction['probability'] * 100

        html += f"""
            <div class="top-prediction">
                <div class="top-prediction-species">{species_name}</div>
                <div class="top-prediction-probability">{probability:.1f}%</div>
            </div>
        """

    if other_predictions:
        html += """
            <div class="other-predictions">
                <h3>Autres possibilités :</h3>
        """

        for pred in other_predictions:
            species_name = get_display_name(pred['species'])
            probability = pred['probability'] * 100

            html += f"""
                <div class="prediction-item">
                    <span class="prediction-species">{species_name}</span>
                    <span class="prediction-probability">{probability:.1f}%</span>
                </div>
            """

        html += """
            </div>
        """

    html += """
            <div class="footer">
                <p>Cette notification a été générée automatiquement par votre système de détection d'oiseaux.</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html
