"""
FastAPI route definitions for DataMind AI.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import shutil
import os
import uuid
import json

from database.session import get_db
from database.models import User, Dataset, AnalysisHistory, MLModel
from database.schemas import (
    DatasetResponse, DatasetPreview, AnalyzeRequest, AnalysisResponse,
    ChatRequest, ChatResponse, VisualizeRequest, VisualizationResponse,
    TrainModelRequest, ModelResponse, ReportResponse
)
from auth import get_current_user
from config import settings
from tools.analysis_tools import load_dataset, get_dataset_overview, clean_dataset, perform_eda, generate_chart_data
from tools.ml_tools import train_classification_model, train_regression_model, train_clustering_model, save_model

router = APIRouter()

# ─── Datasets ──────────────────────────────────────────────────

@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a new dataset."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls", ".json"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{ext}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = load_dataset(file_path)
        overview = get_dataset_overview(df)
        
        dataset = Dataset(
            id=file_id,
            name=file.filename,
            original_filename=file.filename,
            file_path=file_path,
            file_type=ext[1:],
            file_size=os.path.getsize(file_path),
            row_count=overview["shape"]["rows"],
            column_count=overview["shape"]["columns"],
            columns_metadata=overview["columns"],
            owner_id=current_user.id,
            status="ready"
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        return dataset
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", response_model=List[DatasetResponse])
def get_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all datasets for the current user."""
    return db.query(Dataset).filter(Dataset.owner_id == current_user.id).all()


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a preview of the dataset contents."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.owner_id == current_user.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        df = load_dataset(dataset.file_path)
        head_data = df.head(10).replace({float('nan'): None}).to_dict(orient="records")
        missing = df.isnull().sum().to_dict()
        
        return DatasetPreview(
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            shape=list(df.shape),
            head=head_data,
            missing_values=missing,
            statistics=df.describe().round(2).to_dict() if not df.empty else {}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Analysis ──────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_dataset(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Trigger automated analysis on a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id, Dataset.owner_id == current_user.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    analysis = AnalysisHistory(
        dataset_id=dataset.id,
        analysis_type=request.analysis_type,
        status="running",
        parameters=request.parameters
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    # In a real app, this would be a Celery task. For now, we simulate background work.
    def run_analysis_task(analysis_id: str, file_path: str):
        # We need a new DB session for the background task
        from database.session import SessionLocal
        bg_db = SessionLocal()
        try:
            db_analysis = bg_db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
            
            df = load_dataset(file_path)
            
            # Step 1: Clean
            clean_res = clean_dataset(df)
            cleaned_df = clean_res["dataframe"]
            
            # Step 2: EDA
            eda_res = perform_eda(cleaned_df)
            
            # We would invoke CrewAI here, but passing the pandas operations results directly for now
            # To avoid huge LLM wait times in this template
            
            db_analysis.result = {
                "cleaning": clean_res["report"],
                "eda": eda_res
            }
            db_analysis.status = "completed"
            bg_db.commit()
        except Exception as e:
            db_analysis.status = "failed"
            db_analysis.summary = str(e)
            bg_db.commit()
        finally:
            bg_db.close()
            
    background_tasks.add_task(run_analysis_task, analysis.id, dataset.file_path)
    
    return analysis

# ─── Visualizations ────────────────────────────────────────────

@router.post("/visualize", response_model=VisualizationResponse)
def generate_visualization(
    request: VisualizeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate a specific visualization."""
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id, Dataset.owner_id == current_user.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    df = load_dataset(dataset.file_path)
    chart_data = generate_chart_data(
        df, 
        request.chart_type, 
        request.x_column, 
        request.y_column, 
        request.color_column
    )
    
    return VisualizationResponse(
        chart_type=request.chart_type,
        chart_data=chart_data,
        insights=f"Generated {request.chart_type} chart successfully."
    )

# ─── ML Models ─────────────────────────────────────────────────

@router.post("/train-model", response_model=ModelResponse)
def train_ml_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Train a machine learning model."""
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id, Dataset.owner_id == current_user.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    model_record = MLModel(
        name=f"{request.model_type.capitalize()} on {dataset.name}",
        dataset_id=dataset.id,
        model_type=request.model_type,
        algorithm=request.algorithm or "auto",
        target_column=request.target_column,
        feature_columns=request.feature_columns,
        hyperparameters=request.hyperparameters,
        status="training"
    )
    db.add(model_record)
    db.commit()
    db.refresh(model_record)
    
    def run_training_task(model_id: str, file_path: str, req: TrainModelRequest):
        from database.session import SessionLocal
        bg_db = SessionLocal()
        try:
            db_model = bg_db.query(MLModel).filter(MLModel.id == model_id).first()
            df = load_dataset(file_path)
            
            result = None
            if req.model_type == "classification":
                result = train_classification_model(df, req.target_column, req.algorithm or "random_forest", req.feature_columns, req.hyperparameters)
            elif req.model_type == "regression":
                result = train_regression_model(df, req.target_column, req.algorithm or "random_forest", req.feature_columns, req.hyperparameters)
            elif req.model_type == "clustering":
                result = train_clustering_model(df, req.algorithm or "kmeans", req.feature_columns, req.hyperparameters)
                
            if result and "error" not in result:
                db_model.metrics = result["metrics"]
                db_model.algorithm = result["algorithm"]
                db_model.status = "ready"
                
                # Save model file
                model_dir = os.path.join(settings.UPLOAD_DIR, "models")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{model_id}.pkl")
                save_model(result, model_path)
                db_model.model_path = model_path
            else:
                db_model.status = "failed"
                
            bg_db.commit()
        except Exception as e:
            db_model.status = "failed"
            bg_db.commit()
        finally:
            bg_db.close()
            
    background_tasks.add_task(run_training_task, model_record.id, dataset.file_path, request)
    
    return model_record
