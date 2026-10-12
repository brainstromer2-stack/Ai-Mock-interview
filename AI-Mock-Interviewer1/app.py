import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Mock Interviewer", page_icon="🎙️", layout="wide")

QUESTION_BANK = {
    "General HR": [
        "Tell me about yourself.",
        "What is one strength you would bring to this role? Give an example.",
        "Describe a challenge you faced and how you handled it.",
        "Why are you interested in this role?",
        "What skill are you currently working to improve?"
    ],
    "Software Engineering": [
        "Tell me about a project you built and the problem it solved.",
        "How would you explain the difference between a list and a tuple in Python?",
        "Describe a difficult bug you fixed. How did you debug it?",
        "How would you design an API for a simple task-management application?",
        "What do you do to make code reliable and maintainable?"
    ],
    "Data Science": [
        "Describe a data project and the impact of your findings.",
        "How do you handle missing or inconsistent data?",
        "Explain overfitting and how you would reduce it.",
        "How would you choose a metric for an imbalanced classification problem?",
        "How do you communicate technical results to non-technical stakeholders?"
    ],
    "Sales and Customer Success": [
        "How would you learn a customer's needs before recommending a solution?",
        "Tell me about a time you handled a difficult customer.",
        "How would you respond to a prospect who says the product is too expensive?",
        "How do you prioritize several customer requests?",
        "How would you measure success in this role?"
    ],
}

def get_posture_feedback(image_bytes):
    """Estimate a few visible pose-alignment cues from a single webcam image."""
    try:
        import cv2
        import mediapipe as mp
        import numpy as np

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"status": "Image could not be read", "notes": []}

        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not result.pose_landmarks:
            return {
                "status": "Person not clearly detected",
                "notes": ["Try sitting in good lighting with your head and shoulders visible."]
            }

        lm = result.pose_landmarks.landmark
        P = mp.solutions.pose.PoseLandmark
        left_shoulder, right_shoulder = lm[P.LEFT_SHOULDER.value], lm[P.RIGHT_SHOULDER.value]
        left_ear, right_ear = lm[P.LEFT_EAR.value], lm[P.RIGHT_EAR.value]
        notes = []

        if min(left_shoulder.visibility, right_shoulder.visibility) < 0.45:
            notes.append("Both shoulders were not clearly visible; reposition the camera.")
        else:
            shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
            if shoulder_tilt > 0.06:
                notes.append("Your shoulders appear uneven in this frame. Try a relaxed, balanced sitting position.")
            else:
                notes.append("Your shoulders appear reasonably level in this frame.")

        if min(left_ear.visibility, right_ear.visibility) > 0.45 and min(left_shoulder.visibility, right_shoulder.visibility) > 0.45:
            ear_mid_x = (left_ear.x + right_ear.x) / 2
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            offset = abs(ear_mid_x - shoulder_mid_x)
            if offset > 0.09:
                notes.append("Your head appears shifted sideways relative to your shoulders. Try facing the camera naturally.")
            else:
                notes.append("Your head appears roughly centered over your shoulders.")
        else:
            notes.append("Head alignment could not be estimated reliably from this image.")

        return {"status": "Pose detected", "notes": notes}
    except ImportError:
        return {
            "status": "Posture analysis dependencies unavailable",
            "notes": ["Install the requirements in requirements.txt. On unsupported Python versions, use Python 3.10 or 3.11."]
        }
    except Exception as exc:
        return {"status": "Posture analysis unavailable", "notes": [f"Could not analyze image: {exc}"]}


def transcribe_audio(audio_bytes):
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio), None
    except Exception as exc:
        return None, str(exc)


def evaluate_answer(question, answer, role):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                input=[
                    {"role": "system", "content": (
                        "You are a constructive mock interview coach. Assess only the supplied answer, "
                        "not the person's worth or personality. Be specific, fair, and concise. "
                        "Return: score out of 10, what worked, what to improve, a suggested answer structure, "
                        "and one follow-up practice question. Do not invent experience for the candidate."
                    )},
                    {"role": "user", "content": f"Role/category: {role}\nQuestion: {question}\nAnswer: {answer}"}
                ]
            )
            return response.output_text
        except Exception as exc:
            st.warning(f"AI feedback was unavailable, so a basic review is shown. Details: {exc}")

    words = answer.split()
    word_count = len(words)
    has_example = any(x in answer.lower() for x in ["for example", "for instance", "when i", "we built", "i led", "i improved"])
    strengths = []
    improvements = []
    if word_count >= 40:
        strengths.append("You provided enough detail to review.")
    else:
        improvements.append("Add more detail; aim for a clear answer with a specific example.")
    if has_example:
        strengths.append("You included a concrete example or experience cue.")
    else:
        improvements.append("Include a real example, ideally describing the situation, your action, and the result.")
    if any(term in answer.lower() for term in ["result", "improved", "reduced", "increased", "learned", "impact"]):
        strengths.append("You mentioned an outcome, impact, or learning.")
    else:
        improvements.append("Explain the outcome or what you learned.")
    if not strengths:
        strengths.append("You made an attempt and have a starting point to build on.")
    return (
        "### Answer review (basic rubric)\n"
        f"**Estimated completeness: {min(10, max(2, round(word_count / 12)))} / 10**\n\n"
        f"**What worked**\n" + "\n".join(f"- {x}" for x in strengths) +
        "\n\n**What to improve**\n" + "\n".join(f"- {x}" for x in improvements) +
        "\n\n**Suggested structure**\n"
        "1. State your main point.\n2. Give a real example using Situation, Task, Action, Result (STAR).\n"
        "3. Finish with the outcome and what you learned.\n\n"
        "**Practice tip:** Keep the answer focused, use first-person ownership, and avoid claiming results you cannot support."
    )


st.title("🎙️ AI Mock Interviewer")
st.caption("Practice answers, review visible posture cues, and get actionable coaching.")

with st.sidebar:
    st.header("Interview setup")
    role = st.selectbox("Interview category", list(QUESTION_BANK.keys()))
    candidate_name = st.text_input("Name (optional)")
    st.info("Your browser may ask permission to use the microphone and camera. Audio transcription uses Google's online speech recognition service.")

if "role" not in st.session_state or st.session_state.role != role:
    st.session_state.role = role
    st.session_state.question_index = 0
    st.session_state.answers = []
    st.session_state.reviews = []

questions = QUESTION_BANK[role]
idx = st.session_state.question_index

tab_interview, tab_posture, tab_review = st.tabs(["Interview", "Posture check", "Final review"])

with tab_interview:
    st.subheader(f"Question {min(idx + 1, len(questions))} of {len(questions)}")
    if idx < len(questions):
        question = questions[idx]
        st.markdown(f"### {question}")
        st.write("Type your answer or record it below. You can edit the transcript before submitting.")
        typed_answer = st.text_area("Your answer", key=f"answer_{role}_{idx}", height=150)
        audio = st.audio_input("Record your answer")
        transcript = ""
        if audio is not None and st.button("Transcribe recording", key=f"transcribe_{idx}"):
            with st.spinner("Transcribing audio..."):
                transcript, error = transcribe_audio(audio.getvalue())
            if error:
                st.error(f"Transcription failed: {error}")
            else:
                st.session_state[f"transcript_{idx}"] = transcript
                st.success("Transcription ready. Copy or edit it in the answer box above.")
                st.code(transcript)
        if f"transcript_{idx}" in st.session_state:
            st.caption("Latest transcript (copy it into the answer box to submit):")
            st.write(st.session_state[f"transcript_{idx}"])

        if st.button("Submit answer and get feedback", type="primary", key=f"submit_{idx}"):
            final_answer = typed_answer.strip() or st.session_state.get(f"transcript_{idx}", "").strip()
            if not final_answer:
                st.error("Please type an answer or record and transcribe one before submitting.")
            else:
                with st.spinner("Reviewing your answer..."):
                    review = evaluate_answer(question, final_answer, role)
                record = {"question": question, "answer": final_answer, "review": review}
                if len(st.session_state.answers) <= idx:
                    st.session_state.answers.append(record)
                    st.session_state.reviews.append(review)
                else:
                    st.session_state.answers[idx] = record
                    st.session_state.reviews[idx] = review
                st.session_state[f"submitted_{idx}"] = True
        if st.session_state.get(f"submitted_{idx}"):
            st.markdown(st.session_state.answers[idx]["review"])
            if st.button("Next question →", key=f"next_{idx}"):
                st.session_state.question_index = min(idx + 1, len(questions))
                st.rerun()
    else:
        st.success("You've reached the end of this interview.")
        st.write(f"Completed answers: {len(st.session_state.answers)} / {len(questions)}")
        if st.button("Restart interview"):
            st.session_state.question_index = 0
            st.session_state.answers = []
            st.session_state.reviews = []
            st.rerun()

with tab_posture:
    st.subheader("Webcam posture check")
    st.write("Capture a still image with your head and shoulders visible. This is a limited visual cue check, not a medical or personality assessment.")
    photo = st.camera_input("Take a posture snapshot")
    if photo is not None:
        posture = get_posture_feedback(photo.getvalue())
        st.write(f"**Status:** {posture['status']}")
        for note in posture["notes"]:
            st.write(f"- {note}")
        st.caption("Lighting, camera angle, clothing, and body position can affect these estimates.")

with tab_review:
    st.subheader("Overall interview review")
    records = st.session_state.answers
    if not records:
        st.info("Submit at least one answer to see your review here.")
    else:
        st.write(f"**Interview category:** {role}")
        if candidate_name.strip():
            st.write(f"**Candidate:** {candidate_name.strip()}")
        st.write(f"**Questions answered:** {len(records)} / {len(questions)}")
        st.markdown("### Answers and feedback")
        for i, record in enumerate(records, start=1):
            with st.expander(f"Question {i}: {record['question']}"):
                st.markdown("**Your answer**")
                st.write(record["answer"])
                st.markdown(record["review"])
        report = "\n\n".join(
            f"Question {i}: {r['question']}\nAnswer: {r['answer']}\nFeedback:\n{r['review']}"
            for i, r in enumerate(records, start=1)
        )
        st.download_button("Download interview report (.txt)", report, file_name="mock_interview_report.txt", mime="text/plain")
        st.caption("Scores and feedback are practice guidance, not a hiring decision or objective measure of ability.")
