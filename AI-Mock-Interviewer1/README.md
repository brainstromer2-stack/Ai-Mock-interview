# AI Mock Interviewer

A Streamlit practice app that asks role-based interview questions, records spoken answers, transcribes audio, provides answer feedback, and checks a webcam snapshot for a few visible posture-alignment cues.

## Features

- Role-based question sets: General HR, Software Engineering, Data Science, Sales/Customer Success
- Typed answers or microphone recording
- Speech-to-text transcription through SpeechRecognition's Google Web Speech endpoint (internet required)
- Optional AI-generated feedback using the OpenAI API
- Basic offline answer rubric if no API key is configured
- Webcam snapshot posture cues using MediaPipe Pose
- Overall review and downloadable report

## Requirements

Use Python 3.10 or 3.11 for the smoothest compatibility with computer-vision dependencies. A working camera/microphone and internet connection are needed for camera/audio features.

## Run locally

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

If PowerShell blocks activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py`.

### Configure AI feedback (optional)

Open `.env` and set `OPENAI_API_KEY` to your own API key. Never commit `.env` or publish your API key. Without a key, the app uses a simple rule-based rubric.

## Push to GitHub

1. Extract this project folder.
2. Copy its files into your repository `AI-Mock-Interviewer1` (or replace matching starter files after backing them up).
3. Open a terminal in the repository folder.
4. Run:

```bash
git add app.py requirements.txt README.md .env.example .gitignore
git commit -m "Build AI mock interview practice app"
git push origin main
```

If your default branch is not `main`, replace `main` with your branch name.

## Deploy on Streamlit Community Cloud

1. Push the files to GitHub.
2. Create a new app at https://share.streamlit.io/ and select the repository, branch, and `app.py`.
3. Add `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in the app's Secrets settings if you want AI feedback.
4. Deploy. Browser permission and hosting limitations may affect camera and audio functionality.

## Notes and limitations

- The posture module analyzes a single still image, not continuous movement. It estimates shoulder level and head centering only when landmarks are visible.
- Webcam analysis can be affected by camera angle, lighting, occlusion, and accessibility needs. It should not be used to infer confidence, personality, honesty, or employability.
- Speech transcription uses an online service, so audio is sent to that service for processing. Do not record sensitive information.
- AI feedback depends on the configured model and may be incorrect; treat it as practice guidance.
- Do not upload API keys, audio recordings, or personal data to a public repository.
