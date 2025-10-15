# Synthetic Data Generator using TabularGAN and FastAPI

This project, `synthetic_data_generator2`, provides a simple web API to generate synthetic tabular data using a Generative Adversarial Network (GAN). Users can upload their own numeric-only CSV data to train the model and then generate new, artificial data samples that mimic the statistical properties of the original data.

The backend is built with FastAPI and the GAN is implemented using PyTorch.

---

## Features

-   **Train Endpoint (`/train`):** Upload a CSV file to train the GAN model on your data.
-   **Generate Endpoint (`/generate`):** Generate a specified number of synthetic data samples from the trained model.
-   **Interactive API Docs:** Powered by FastAPI for easy testing directly in the browser.

---

## Setup & Installation

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.8+

### Steps

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    ```

2.  **Navigate to the Project Directory**
    ```bash
    cd synthetic_data_generator2
    ```

3.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    ```

4.  **Activate the Virtual Environment**
    -   On Windows (PowerShell):
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

5.  **Install Required Packages**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run the Application

1.  With your virtual environment activated, start the FastAPI server using `uvicorn`:
    ```bash
    python -m uvicorn main:app --reload
    ```

2.  Open your web browser and navigate to the interactive API documentation:
    [**http://127.0.0.1:8000/docs**](http://127.0.0.1:8000/docs)

---

## How to Use the API

Once the server is running, use the `/docs` page to interact with the API.

### 1. Initial API Documentation

Here's how the interactive API documentation looks when you first open it:

![FastAPI Docs Initial View](C:\Users\khush\OneDrive\Pictures\Get1.png)

### 2. Train the Model

To train the GAN, expand the `POST /train` endpoint, click "Try it out", and upload your **numeric-only** CSV file (e.g., `sample_data.csv`).

![FastAPI Train Endpoint Input](C:\Users\khush\OneDrive\Pictures\Get2.png)

After successful execution, you'll receive a confirmation:

![FastAPI Train Endpoint Output](C:\Users\khush\OneDrive\Pictures\Generate1.png)

### 3. Generate Data

After the model is trained, you can generate new synthetic data. Expand the `GET /generate` endpoint, click "Try it out", and specify the number of samples (e.g., 100).

![FastAPI Generate Endpoint Input](C:\Users\khush\OneDrive\Pictures\Generate2.png)



---
