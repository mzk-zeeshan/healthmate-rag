# HealthMate Project 🩺

HealthMate is a personal health assistant application powered by Google Gemini AI. It helps users manage their health by providing an intelligent consultation bot and a medical record analysis system.

## Features

-   **User Authentication**: Secure Sign Up and Login system.
-   **CuraBot (Consultation Bot)**: An AI-powered chatbot that acts as a medical assistant. Users can describe symptoms and get insights on possible conditions and treatments.
-   **Medical Record Bot (RAG System)**:
    -   Upload PDF medical reports.
    -   View extracted text from reports.
    -   Chat with your medical reports to get specific answers based on the document content.
-   **Data Privacy**: User data and files are stored locally and securely.

## Tech Stack

-   **Frontend**: Streamlit
-   **AI Model**: Google Gemini 2.0 Flash
-   **Embeddings**: Google Generative AI Embeddings
-   **Vector Store**: FAISS
-   **Database**: SQLite

## Installation

1.  **Clone the repository** (or download the files).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up Environment Variables**:
    -   Create a `.env` file in the root directory.
    -   Add your Google API Key:
        ```
        GOOGLE_API_KEY=your_api_key_here
        ```

## Usage

Run the application using Streamlit:

```bash
streamlit run main.py
```

The app will open in your default web browser.
