# AI Student Matching System

An intelligent study partner recommendation system that uses Natural Language Processing (NLP) to connect students based on their subjects, availability, and personal interests.

## Features
- **Hard Constraints Filtering:** Matches users based on shared subjects and overlapping free time.
- **AI-Powered Similarity:** Uses `Sentence-Transformers` (all-MiniLM-L6-v2) to calculate personality and hobby compatibility.
- **Interactive UI:** A clean, user-friendly web interface built with Streamlit.
- **Optimized Matching:** Utilizes Cosine Similarity and KNN for efficient results.

## Tech Stack
- **Language:** Python
- **AI/ML:** Sentence-Transformers, Scikit-learn
- **Data:** Pandas, NumPy
- **Interface:** Streamlit

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/AI-WFox/AI-Student-Matching.git](https://github.com/AI-WFox/AI-Student-Matching.git)

   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   
   ```

## Docker

1. Build the image:
```bash
docker build -t ai-student-matching:latest .
```

2. Run the image (port 8080):
```bash
# We mount a local folder to the container's '/app/data'
docker run -p 8080:8501 -v $(pwd)/data:/app/data ai-student-matching:latest
```

3. Docker compose:
```bash
docker compose up
```

## Project Structure

* `app.py`: The main Streamlit web application.
* `main.py`: Core matching logic and CLI version.
* `user_data.csv`: Sample dataset for testing.
