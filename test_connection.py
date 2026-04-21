import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# טעינת המשתנים מקובץ ה-env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key and api_key.startswith("sk-"):
    print("✅ מפתח ה-API נמצא בפורמט הנכון!")
    try:
        # ניסיון ליצור אובייקט של Embeddings (זה מבצע קריאה קטנה ל-API)
        embeddings = OpenAIEmbeddings()
        print("✅ הצלחנו להתחבר ל-OpenAI בהצלחה!")
    except Exception as e:
        print(f"❌ שגיאה בהתחברות: {e}")
else:
    print("❌ המפתח לא נמצא או שהוא לא תקין. בדקי את קובץ ה-.env שלך.")