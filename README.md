# Intelligent Accent Analysis Tool

## Overview

https://accent-tool-project.streamlit.app/

This project is a Streamlit web app designed to analyze English speech accents using audio from YouTube videos. The tool automatically downloads audio, transcribes it with OpenAI's Whisper model, and performs accent classification using a fine-tuned model based on SpeechBrain.

_To Note: american accent seems to be the hardest to detect by the model, it might fail to recognize the accent entirely_

---

## Approach

### Accent Detection Pipeline

1. **Audio Source**:

   * User provides a **YouTube URL** containing English speech.
   * Audio is extracted using `yt-dlp`.

2. **Preprocessing**:

   * Audio is converted and cleaned using **FFmpeg**.

3. **Transcription**:

   * Transcription is performed with **OpenAI Whisper** to validate English speech.

4. **Accent Classification**:

   * The extracted audio is passed through a **fine-tuned SpeechBrain model** trained to distinguish different English accents.

5. **Result Display**:

   * The predicted accent is displayed along with metadata and optional transcription.

---



## Challenges Solved

### ✅ SentencePiece Build Errors (Streamlit Cloud)

* Streamlit Cloud does not use `sudo` or `apt` directly. Instead, we use an `packages.txt` to list dependencies such as `ffmpeg`.

### ✅ Python Version Compatibility

* The app uses **Python 3.10** for compatibility with `sentencepiece`. Python 3.13 breaks builds due to missing build tools.

### ✅ HTTP 403 Video Download Errors

* Solved by setting a **browser-like User-Agent** in `yt-dlp` to bypass download restrictions on YouTube.

### ✅ Whisper Build Fails

* Added proper installation tools like `openai-whisper` and ensured the Streamlit app had access to C++ build tools.

---

## Limitations

* **Input Restricted to YouTube URLs**: Only YouTube links have been tested currently.
* **Repeated URLs May Fail**: Re-downloading the same video can cause caching or quota issues, specifically overwriting permission of video file downloaded asked in terminal would need to be automated somehow.
* **Resource Constraints**:

  * Streamlit Community Cloud limits CPU and RAM.
  * Longer or high-quality videos may timeout or be truncated.

---

## Prerequisites (For Local Use)

Ensure you have:

* **FFmpeg** installed and on your system PATH:

  * macOS: `brew install ffmpeg`
  * Ubuntu/Debian: `sudo apt-get update && sudo apt-get install ffmpeg`
  * Windows: Install from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
* **Python 3.8+** with virtualenv or conda.

---

## File Structure

```
accent_tool_project/
├── app.py
├── requirements.txt
├── packages.txt      # for Streamlit Cloud package installs
└── README.md
```

This ensures Streamlit Cloud installs essential system packages:

* **FFmpeg** (audio handling)
* **pkg-config**, **cmake** (C++ build tools)
* **libsentencepiece-dev** (required by `sentencepiece` Python module)

---

## Tests

For now minimal testing has been done because of time constraint, thorough testing must be done and results stored in a database to evaluate model.

- Australian accent video: https://www.youtube.com/watch?v=UByw5C2d1IU

### Results 

Predicted Accent: australia

Confidence: 100.0%

Explanation

Normalized among English accents, this sample best matches australia with 100.0% confidence.

The classifier detected characteristics common to australia-accented English speech.

- British accent video: https://www.youtube.com/shorts/mFNZ8-Tbi30

Predicted Accent: england

Confidence: 100.0%

Explanation

Normalized among English accents, this sample best matches england with 100.0% confidence.

The classifier detected characteristics common to england-accented English speech.

## Goals Achieved ✅

* Practicality taken into account.
* Built a working end-to-end **accent detection web app**.
* Solved build and compatibility issues on **Streamlit Cloud**.
* Successfully integrated **Whisper**, **SpeechBrain**, and **yt-dlp**.
* Documented limitations, solutions, and deployment strategy.

---

## Time taken

2-3 hours taken. From understanding problem, to coding solution, testing locally and deployment, solving problems along the way. Also readme for documentation.


## Future Improvements

* Thorough testing and evaluation.
* Support audio file uploads.
* Enable multi-accent classification and scoring.
* Add retry logic and cache handling for repeated URLs.
* Migrate to GPU-backed hosting for improved performance.
