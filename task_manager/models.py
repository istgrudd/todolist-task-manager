import csv
import os
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Task:
    """Class representing a single task with all its attributes"""
    PRIORITY_DURATIONS = {
        "Tinggi": 2.0,
        "Sedang": 4.0,
        "Rendah": 1.0
    }
    
    nama: str
    prioritas: str
    deadline: datetime.date
    deskripsi: str = ""
    selesai: bool = False
    tanggal_selesai: Optional[datetime.date] = None
    durasi_aktual: Optional[float] = None
    durasi_estimasi: float = field(init=False)
    waktu_rekomendasi: Optional[datetime] = None

    def __post_init__(self):
        self.durasi_estimasi = self._estimate_initial_duration()
        self._validate_data()
    
    def _estimate_initial_duration(self):
        """Estimate initial task duration based on priority"""
        return self.PRIORITY_DURATIONS.get(self.prioritas, 2.0)

    def _validate_data(self):
        """Validate task data integrity"""
        if self.selesai and not self.tanggal_selesai:
            raise ValueError("Task selesai harus memiliki tanggal_selesai")
        if self.selesai and self.durasi_aktual is None:
            raise ValueError("Task selesai harus memiliki durasi_aktual")
        if not isinstance(self.deadline, date):
            raise ValueError("Deadline harus berupa date object")

    def mark_completed(self, tanggal_selesai: Optional[date] = None, durasi_aktual: Optional[float] = None) -> None:
        """Mark task as completed with required data"""
        if durasi_aktual is None:
            raise ValueError("Durasi aktual harus diisi")
        
        self.selesai = True
        self.tanggal_selesai = tanggal_selesai or datetime.now().date()
        self.durasi_aktual = durasi_aktual
        self._validate_data()

    def to_dict(self) -> Dict:
        """Convert task object to dictionary for CSV storage"""
        return {
            "Nama": self.nama,
            "Deskripsi": self.deskripsi,
            "Prioritas": self.prioritas,
            "Deadline": self.deadline.strftime("%Y-%m-%d"),
            "Selesai": str(self.selesai),
            "Tanggal_Selesai": self.tanggal_selesai.strftime("%Y-%m-%d") if self.tanggal_selesai else "",
            "Durasi_Aktual": str(self.durasi_aktual) if self.durasi_aktual is not None else "",
            "Durasi_Estimasi": str(self.durasi_estimasi),
            "Waktu_Rekomendasi": self.waktu_rekomendasi.strftime("%Y-%m-%d %H:%M") if self.waktu_rekomendasi else ""
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create Task object from dictionary (CSV data) with validation"""
        try:
            # Parse completion status
            selesai = data.get("Selesai", "False").strip().lower() == "true"
            
            # Parse and validate completion date if task is marked completed
            tanggal_selesai = None
            if selesai and data.get("Tanggal_Selesai", "").strip():
                try:
                    tanggal_selesai = datetime.strptime(data["Tanggal_Selesai"].strip(), "%Y-%m-%d").date()
                except ValueError:
                    selesai = False  # Auto-correct if date is invalid
            
            # If marked completed but no valid date, auto-correct to not completed
            if selesai and not tanggal_selesai:
                selesai = False
            
            # Create task instance
            task = cls(
                nama=data["Nama"].strip(),
                deskripsi=data.get("Deskripsi", "").strip(),
                prioritas=data["Prioritas"].strip(),
                deadline=datetime.strptime(data["Deadline"].strip(), "%Y-%m-%d").date(),
                selesai=selesai,
                tanggal_selesai=tanggal_selesai
            )
            
            # Parse duration if available
            if selesai and data.get("Durasi_Aktual", "").strip():
                try:
                    task.durasi_aktual = float(data["Durasi_Aktual"].strip())
                except ValueError:
                    task.durasi_aktual = task.durasi_estimasi
            
            # Parse recommended time if available
            if data.get("Waktu_Rekomendasi", "").strip():
                try:
                    task.waktu_rekomendasi = datetime.strptime(data["Waktu_Rekomendasi"].strip(), "%Y-%m-%d %H:%M")
                except ValueError:
                    pass  # Skip if invalid format
            
            return task
            
        except (ValueError, KeyError) as e:
            # Create minimal valid task if critical fields are missing
            try:
                return cls(
                    nama=data.get("Nama", "Task Recovery").strip(),
                    deskripsi=data.get("Deskripsi", "").strip(),
                    prioritas=data.get("Prioritas", "Sedang").strip(),
                    deadline=datetime.now().date(),
                    selesai=False
                )
            except Exception:
                raise ValueError(f"Invalid task data that couldn't be recovered: {str(e)}")


class TaskManager:
    """Main class for managing tasks and their operations"""
    
    def __init__(self):
        self.tasks: List[Task] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._load_from_csv()

    def _load_from_csv(self) -> None:
        """Load tasks from CSV file with error recovery"""
        if not os.path.exists("tugas.csv"):
            self.tasks = []
            return
            
        self.tasks = []
        with open("tugas.csv", "r", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    task = Task.from_dict(row)
                    self.tasks.append(task)
                except ValueError as e:
                    print(f"Warning: Skipping invalid task - {e}")
                    continue

    def save_to_csv(self):
        """Save tasks to CSV file with data validation"""
        try:
            with open("tugas.csv", "w", newline="", encoding='utf-8') as f:
                fieldnames = [
                    "Nama", "Deskripsi", "Prioritas", "Deadline", 
                    "Selesai", "Tanggal_Selesai", "Durasi_Aktual", 
                    "Durasi_Estimasi", "Waktu_Rekomendasi"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for task in self.tasks:
                    try:
                        # Ensure completed tasks have valid data
                        if task.selesai:
                            if not task.tanggal_selesai:
                                task.selesai = False
                            if task.durasi_aktual is None:
                                task.durasi_aktual = task.durasi_estimasi
                        
                        writer.writerow(task.to_dict())
                    except Exception as e:
                        print(f"Error saving task {task.nama}: {e}")
                        continue
                        
            return True
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False

    def get_valid_completed_tasks(self) -> List[Task]:
        """Get list of properly completed tasks (with all required data)"""
        return [
            t for t in self.tasks 
            if t.selesai and t.tanggal_selesai and t.durasi_aktual is not None
        ]

    def add_task(self, nama: str, deskripsi: str, prioritas: str, deadline: str) -> tuple:
        """Add new task with validation and time recommendation"""
        try:
            deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
            task = Task(nama, prioritas, deadline_date, deskripsi=deskripsi)
            self._generate_time_recommendation(task)
            self.tasks.append(task)
            self.save_to_csv()
            return True, task
        except ValueError:
            return False, "Format tanggal salah!"

    def _generate_time_recommendation(self, task: Task) -> None:
        """Generate optimal working time recommendation for task"""
        if not self.tasks:
            task.waktu_rekomendasi = datetime.now() + timedelta(hours=1)
            return
        
        task.durasi_estimasi = self._estimate_duration(task)
        task.waktu_rekomendasi = self._find_optimal_time_slot(task)
        self._adjust_for_deadline(task)

    def _estimate_duration(self, new_task: Task) -> float:
        """Estimate duration based on similar completed tasks"""
        completed_tasks = [t for t in self.tasks if t.selesai and t.durasi_aktual]
        
        if not completed_tasks:
            return new_task.durasi_estimasi
            
        texts = [f"{t.nama} {t.deskripsi}" for t in completed_tasks]
        texts.append(f"{new_task.nama} {new_task.deskripsi}")

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        
        similar_indices = np.argsort(similarities)[-3:]
        similar_tasks = [completed_tasks[i] for i in similar_indices]
        
        avg_duration = np.mean([t.durasi_aktual for t in similar_tasks])
        return (avg_duration * 0.7) + (new_task.durasi_estimasi * 0.3)

    def _find_optimal_time_slot(self, task: Task) -> datetime:
        """Find optimal time slot based on user's productivity patterns"""
        completed_tasks = [t for t in self.tasks if t.selesai and t.tanggal_selesai]
        
        if not completed_tasks:
            return datetime.now().replace(
                hour=9, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            
        completion_days = [t.tanggal_selesai.weekday() for t in completed_tasks]
        most_productive_day = Counter(completion_days).most_common(1)[0][0]
        
        today = datetime.now()
        days_ahead = (most_productive_day - today.weekday()) % 7
        if days_ahead <= 0:
            days_ahead += 7
            
        return (today + timedelta(days=days_ahead)).replace(
            hour=10, minute=0, second=0, microsecond=0
        )

    def _adjust_for_deadline(self, task: Task) -> None:
        """Adjust recommendation to ensure it's before deadline"""
        if task.waktu_rekomendasi.date() > task.deadline:
            task.waktu_rekomendasi = datetime.combine(
                task.deadline - timedelta(days=1), 
                datetime.min.time()
            ).replace(hour=14)
            
        if task.waktu_rekomendasi < datetime.now():
            task.waktu_rekomendasi = datetime.now() + timedelta(hours=1)

    def get_active_tasks(self) -> List[Task]:
        """Get list of incomplete tasks"""
        return [t for t in self.tasks if not t.selesai]

    def get_completed_tasks(self) -> List[Task]:
        """Get list of completed tasks"""
        return [t for t in self.tasks if t.selesai]

    def get_tasks_by_deadline(self, days: int = 7) -> Dict[datetime.date, List[Task]]:
        """Get tasks grouped by deadline for the next specified days"""
        today = datetime.now().date()
        calendar = {today + timedelta(days=i): [] for i in range(days)}
        
        for task in self.tasks:
            if task.deadline in calendar:
                calendar[task.deadline].append(task)
                
        return calendar