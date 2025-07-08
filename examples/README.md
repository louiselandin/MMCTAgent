# MMCT Agent Examples

This directory contains example notebooks demonstrating how to use the MMCT Agent framework components.

## 📓 Available Examples

### 1. **image_agent.ipynb**
Demonstrates how to use the `ImageAgent` for image analysis tasks.

**Features:**
- Image analysis using multiple tools (OCR, Object Detection, VIT, RECOG)
- Optional critic agent for improved accuracy
- Configurable tool selection

**Usage:**
```python
from mmct.image_pipeline import ImageAgent, ImageQnaTools

agent = ImageAgent(
    query="describe the image",
    image_path="path/to/image.jpg",
    tools=[ImageQnaTools.OCR, ImageQnaTools.VIT],
    use_critic_agent=True
)
response = await agent()
```

### 2. **video_agent.ipynb**
Shows how to use the `VideoAgent` for video question answering.

**Features:**
- Video retrieval from Azure AI Search index
- Multi-modal analysis using MMCT framework
- Support for Azure Computer Vision integration
- Configurable critic agent

**Usage:**
```python
from mmct.video_pipeline import VideoAgent

agent = VideoAgent(
    query="what is discussed in the video?",
    index_name="video-index",
    top_n=2,
    use_azure_cv_tool=False,
    use_critic_agent=True
)
response = await agent()
```

### 3. **ingestion_pipeline.ipynb**
Demonstrates the video ingestion pipeline for processing videos.

**Features:**
- Video transcription using Whisper or Azure Speech-to-Text
- Frame extraction and chapter generation
- Azure AI Search indexing
- Optional Azure Computer Vision integration

**Usage:**
```python
from mmct.video_pipeline import IngestionPipeline, Languages, TranscriptionServices

pipeline = IngestionPipeline(
    video_path="path/to/video.mp4",
    index_name="video-index",
    transcription_service=TranscriptionServices.WHISPER,
    language=Languages.ENGLISH_INDIA,
    use_azure_computer_vision=False
)
await pipeline()
```

## 🛠️ Setup Requirements

Before running these examples:

1. **Environment Setup**: Create a `.env` file in the root directory with required Azure credentials
2. **Install Dependencies**: Run `pip install -r requirements.txt`
3. **Azure Services**: Ensure you have access to required Azure services (OpenAI, Storage, Search, etc.)

## 📋 Notes

- All examples use `nest_asyncio` to support async operations in Jupyter notebooks
- Update file paths in the examples to match your local setup
- The examples assume you have properly configured Azure services and credentials