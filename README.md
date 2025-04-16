# YouTube Shorts Content Classification

This project applies machine learning techniques to classify YouTube Shorts as either "safe" or "unsafe" based on various features extracted from video content, including transcript analysis, image content classification, and audio transcription.

## Overview

The goal of this project is to classify YouTube Shorts videos into two categories:
- **Safe**: Videos that do not contain toxic, inappropriate, or unsafe content.
- **Unsafe**: Videos that contain any form of toxicity, profanity, violence, or NSFW content.

## Features

1. **Audio Transcription**: Extracts and transcribes audio from YouTube Shorts videos using OpenAI's Whisper model.
2. **Text Analysis**: Evaluates toxicity and profanity in the transcript using pre-trained models.
3. **Image Analysis**: Identifies potentially NSFW content in video frames using a pre-trained NSFW model.
4. **Unsafe Content Detection**: Flags content based on the occurrence of unsafe words like violence, death, and drugs.
5. **Machine Learning Classification**: Trains a Random Forest classifier to classify videos as "safe" or "unsafe" based on extracted features.

## Requirements

Before running this project, ensure you have the following dependencies installed:

```bash
pip install transformers pillow opencv-python ffmpeg-python openai-whisper pandas
```

## Models Used

- **Whisper**: OpenAI's Whisper model for automatic speech recognition (ASR) to transcribe audio.
- **Toxic BERT**: A BERT-based model for detecting toxic text.
- **NSFW Image Detection**: A pre-trained model for classifying images as NSFW or safe.

## Functionality

### 1. **Extract Audio and Transcribe**:
This function extracts audio from video files and transcribes them using the Whisper ASR model.

```python
def extract_audio(video_path):
    # Extracts audio from video file
    ...
```

### 2. **Text Classification**:
The transcript of the video is analyzed for toxicity, profanity, and unsafe content using pre-trained models.

```python
def get_toxicity_score(text):
    # Analyzes the text for toxicity
    ...
```

### 3. **NSFW Image Classification**:
Video frames are extracted at regular intervals and checked for NSFW content.

```python
def get_nsfw_score(frames):
    # Analyzes image frames for NSFW content
    ...
```

### 4. **Video Safety Classification**:
Combines audio transcription, text classification, and image analysis to classify the video as "safe" or "unsafe".

```python
def process_video(video_path, title="", description=""):
    # Process the video and classify as safe or unsafe
    ...
```

### 5. **Train Classifier**:
A Random Forest classifier is trained on labeled data and used to classify new videos. 
**-----Important: Do not run this if you do not have the specified csv file containing the processesed data in the mentioned location----** 

```python
def classify_video(features):
    # Classifies new videos based on trained model
    ...
```

## Example Usage

To process a single video and classify it as "safe" or "unsafe":

```python
features = {
    "toxicity_score": 0.3,
    "nsfw_score": 0.2,
    "profanity_count": 1,
    "violence_score": 0,
    "unsafe_words_count": 2,
}
classification_result = classify_video(features)
print("Video classification:", classification_result)
```

## Data

The project assumes you have a CSV file with information about YouTube Shorts videos. This file should contain columns like `file`, `title`, and `description`.

## Results

The processed features are stored in a CSV file, which includes columns for the following:
- `toxicity_score`
- `nsfw_score`
- `profanity_count`
- `violence_score`
- `unsafe_words_count`
- `safety_score` (Label indicating if the video is "safe" or "unsafe")

## Future Improvements

- Extend the classifier to handle additional content types, such as text overlays or metadata.
- Use more than one advanced models for more accurate NSFW and toxicity classification.
- Create a mobile app to filter the shorts as they are viewed by kids.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README introduces the project clearly and provides key details for users to understand how to use it and what models and techniques are involved. Let me know if you'd like any changes or additional details!
