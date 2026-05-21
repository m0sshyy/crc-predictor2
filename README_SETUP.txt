CRC Streamlit Update

Files included:
- app.py
- requirements.txt
- assets/digestive_system.png

Before running or deploying:
1. Copy your trained model file into this folder:
   crc_xgboost_model.pkl

2. Your GitHub repository should look like this:
   app.py
   requirements.txt
   crc_xgboost_model.pkl
   assets/
      digestive_system.png

3. Run locally:
   streamlit run app.py

4. For Streamlit Community Cloud:
   - Push all files to GitHub.
   - Set main file path as app.py.
   - Ensure requirements.txt is in the repository root.

Important:
This prototype is for academic demonstration only and is not a medical diagnosis tool.
