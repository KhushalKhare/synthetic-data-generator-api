import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from sklearn.preprocessing import StandardScaler
from gan import TabularGAN # This correctly imports your class from gan.py

# --- 1. App and Model Initialization ---
app = FastAPI(title="Tabular GAN for Synthetic Data")
scaler = None
gan_tabular = None

# --- 2. Root Endpoint ---
@app.get("/")
def root():
    return {"message": "Welcome! Use /docs to try the GAN system."}

# --- 3. Training Endpoint (CORRECTED) ---
@app.post("/train")
async def train_gan(file: UploadFile = File(...)):
    global scaler, gan_tabular

    try:
        df = pd.read_csv(file.file)

        # Check for non-numeric columns before scaling
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
            raise HTTPException(status_code=400, detail="CSV file contains non-numeric data. Please provide a file with numbers only.")

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)

        # CORRECTED: Changed 'num_features' to 'input_dim' to match your class
        gan_tabular = TabularGAN(input_dim=data_scaled.shape[1])

        # CORRECTED: Changed '.fit' to '.train' to match your class method
        gan_tabular.train(data_scaled, epochs=500) # Using 500 epochs from your code

        return {"message": "GAN model trained successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {e}")

# --- 4. Generation Endpoint (CORRECTED) ---
@app.get("/generate")
def generate_data(num_samples: int = 100):
    global scaler, gan_tabular

    if gan_tabular is None or scaler is None:
        raise HTTPException(status_code=400, detail="Model has not been trained yet. Please upload data to the /train endpoint first.")

    try:
        # CORRECTED: Changed '.sample' to '.generate' and 'n' to match your class method
        generated_data_scaled = gan_tabular.generate(n=num_samples)

        generated_data_original_scale = scaler.inverse_transform(generated_data_scaled)

        # Use original column names if possible
        try:
            columns = scaler.get_feature_names_out()
        except AttributeError:
            columns = [f"feature_{i+1}" for i in range(generated_data_original_scale.shape[1])]

        df_generated = pd.DataFrame(generated_data_original_scale, columns=columns)
        result = df_generated.to_dict(orient="records")
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during data generation: {e}")
