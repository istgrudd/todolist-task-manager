from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Task


class TimeOptimizer:
    """Optimize time allocation for tasks using various algorithms"""
    
    def __init__(self, tasks: List[Task]):
        self.tasks = tasks
        self.vectorizer = TfidfVectorizer(stop_words="english")
    
    def optimize_schedule(self, new_task: Task) -> Dict:
        """Generate optimal schedule for new task considering existing tasks"""
        # Content-based similarity
        similar_tasks = self._find_similar_tasks(new_task)
        
        # Time optimization
        optimal_time = self._find_optimal_time(new_task, similar_tasks)
        
        # Priority adjustment
        priority_score = self._calculate_priority_score(new_task, similar_tasks)
        
        return {
            'optimal_time': optimal_time,
            'priority_score': priority_score,
            'similar_tasks': similar_tasks
        }
    
    def _find_similar_tasks(self, new_task: Task) -> List[Task]:
        """Find similar tasks using content-based filtering"""
        if not self.tasks:
            return []
            
        texts = [f"{t.nama} {t.deskripsi}" for t in self.tasks]
        texts.append(f"{new_task.nama} {new_task.deskripsi}")
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        
        similar_indices = np.argsort(similarities)[-3:]  # Top 3 similar tasks
        return [self.tasks[i] for i in similar_indices]
    
    def _find_optimal_time(self, new_task: Task, similar_tasks: List[Task]) -> datetime:
        """Find optimal time slot considering similar tasks"""
        if not similar_tasks:
            return datetime.now() + timedelta(hours=1)
            
        # Get completion times of similar tasks
        completion_times = [
            t.waktu_rekomendasi for t in similar_tasks 
            if t.waktu_rekomendasi and t.selesai
        ]
        
        if completion_times:
            avg_hour = np.mean([t.hour for t in completion_times])
            best_hour = min(17, max(9, int(avg_hour)))  # Normalize to work hours
        else:
            best_hour = 10  # Default to 10 AM
            
        # Find next available slot
        now = datetime.now()
        candidate = now.replace(hour=best_hour, minute=0, second=0, microsecond=0)
        
        if candidate < now:
            candidate += timedelta(days=1)
            
        return candidate
    
    def _calculate_priority_score(self, new_task: Task, similar_tasks: List[Task]) -> float:
        """Calculate dynamic priority score based on similar tasks"""
        if not similar_tasks:
            return 0.5  # Neutral score
            
        # Get priorities of similar tasks (convert to numerical)
        priority_map = {"Tinggi": 1.0, "Sedang": 0.66, "Rendah": 0.33}
        similar_priorities = [priority_map.get(t.prioritas, 0.5) for t in similar_tasks]
        
        # Weighted average considering task similarity
        return np.mean(similar_priorities)
