from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, List
from .models import Task


def analyze_productivity_patterns(tasks: List[Task]) -> Dict:
    """Analyze user's productivity patterns with time series and clustering"""
    completed_tasks = [t for t in tasks if t.selesai and t.tanggal_selesai and t.durasi_aktual]
    
    if not completed_tasks:
        return None
        
    # Time Series Analysis
    df = pd.DataFrame({
        'date': [t.tanggal_selesai for t in completed_tasks],
        'duration': [t.durasi_aktual for t in completed_tasks],
        'weekday': [t.tanggal_selesai.weekday() for t in completed_tasks],
        'hour': [t.waktu_rekomendasi.hour if t.waktu_rekomendasi else 12 for t in completed_tasks]
    })
    
    # Productivity by time of day
    hourly_productivity = df.groupby('hour')['duration'].mean()
    
    # Weekly patterns
    weekday_productivity = df.groupby('weekday')['duration'].mean()
    
    # Clustering user behavior
    X = df[['weekday', 'hour', 'duration']].dropna()
    if len(X) >= 3:  # Minimum samples for clustering
        kmeans = KMeans(n_clusters=min(3, len(X)), random_state=42)
        clusters = kmeans.fit_predict(X)
        df.loc[X.index, 'cluster'] = clusters
        cluster_centers = kmeans.cluster_centers_
    else:
        cluster_centers = None
    
    return {
        'hourly_productivity': hourly_productivity.to_dict(),
        'weekday_productivity': weekday_productivity.to_dict(),
        'productivity_clusters': cluster_centers,
        'raw_data': df
    }


def predict_task_delay(task: Task, tasks: List[Task]) -> float:
    """Predict probability of task delay using machine learning"""
    completed_tasks = [t for t in tasks if t.selesai and t.durasi_aktual and t.tanggal_selesai]
    
    if len(completed_tasks) < 5:  # Minimum number of completed tasks needed
        return 0.0  # Default to no delay if not enough data
        
    # Prepare features and target
    X = []
    y = []
    
    for t in completed_tasks:
        days_to_deadline = (t.deadline - t.tanggal_selesai).days
        was_delayed = 1 if days_to_deadline < 0 else 0
        X.append([
            t.durasi_estimasi,
            t.durasi_aktual,
            days_to_deadline,
            t.prioritas == "Tinggi",
            t.prioritas == "Sedang"
        ])
        y.append(was_delayed)
    
    try:
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict for current task
        task_features = [
            task.durasi_estimasi,
            task.durasi_estimasi,  # Using estimate since actual not available
            (task.deadline - datetime.now().date()).days,
            task.prioritas == "Tinggi",
            task.prioritas == "Sedang"
        ]
        
        proba = model.predict_proba([task_features])
        return proba[0][1] if proba.shape[1] > 1 else 0.0  # Return probability of delay or 0 if model can't predict
    except Exception:
        return 0.0  # Return 0 if any error occurs
