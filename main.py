import streamlit as st
from datetime import datetime, date, time
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from task_manager.models import TaskManager
from task_manager.analytics import analyze_productivity_patterns, predict_task_delay
from task_manager.recommendations import TimeOptimizer
import uuid


def init_session_state():
    """Initialize session state variables"""
    if 'task_manager' not in st.session_state:
        st.session_state.task_manager = TaskManager()
    if 'productivity_data' not in st.session_state:
        st.session_state.productivity_data = None


def setup_page():
    """Configure page settings"""
    st.set_page_config(
        page_title="Sistem Manajemen Tugas Cerdas",
        page_icon="📋",
        layout="wide"
    )
    st.title("📋 Sistem Manajemen Tugas Cerdas")


def show_main_menu():
    """Display main navigation menu"""
    menu = st.sidebar.selectbox(
        "Menu",
        ["Tambah Tugas", "Daftar Tugas", "Kalender", "Statistik", "Rekomendasi", "Analisis Produktivitas"]
    )
    
    if menu == "Tambah Tugas":
        show_add_task()
    elif menu == "Daftar Tugas":
        show_task_list()
    elif menu == "Kalender":
        show_calendar()
    elif menu == "Statistik":
        show_statistics()
    elif menu == "Rekomendasi":
        show_recommendations()
    elif menu == "Analisis Produktivitas":
        show_productivity_analysis()


def show_add_task():
    """Display task addition form"""
    st.header("➕ Tambah Tugas Baru")
    
    with st.form("task_form"):
        nama = st.text_input("Nama Tugas*")
        deskripsi = st.text_area("Deskripsi Tugas (opsional)")
        prioritas = st.selectbox("Prioritas*", ["Tinggi", "Sedang", "Rendah"])
        deadline = st.date_input("Deadline*", min_value=date.today())
        
        submitted = st.form_submit_button("Simpan Tugas")
        
        if submitted:
            if not nama:
                st.error("Nama tugas wajib diisi!")
            else:
                success, result = st.session_state.task_manager.add_task(
                    nama=nama,
                    deskripsi=deskripsi,
                    prioritas=prioritas,
                    deadline=deadline.strftime("%Y-%m-%d")
                )
                
                if success:
                    task = result
                    st.success("Tugas berhasil ditambahkan!")
                    if task.waktu_rekomendasi:
                        st.info(
                            f"💡 Rekomendasi: Kerjakan tugas ini pada {task.waktu_rekomendasi.strftime('%A, %d %B %Y pukul %H:%M')}\n"
                            f"⏱ Estimasi durasi: {task.durasi_estimasi:.1f} jam"
                        )
                    
                    # Show delay prediction
                    delay_prob = predict_task_delay(task, st.session_state.task_manager.tasks)
                    if delay_prob > 0.3:
                        st.warning(f"⚠️ Potensi keterlambatan: {delay_prob*100:.1f}%")
                else:
                    st.error(result)


def show_task_list():
    st.header("📝 Daftar Tugas")
    
    # Initialize session state variables
    if 'completing_task' not in st.session_state:
        st.session_state.completing_task = None
    if 'editing_task' not in st.session_state:
        st.session_state.editing_task = None
    if 'edit_form_key' not in st.session_state:
        st.session_state.edit_form_key = str(uuid.uuid4())

    # [Keep your existing filter and task display code here]
    # Filter tasks
    col1, col2 = st.columns(2)
    with col1:
        show_completed = st.checkbox("Tampilkan tugas selesai", value=True)
    with col2:
        show_incomplete = st.checkbox("Tampilkan tugas belum selesai", value=True)
    
    search_query = st.text_input("Cari tugas berdasarkan nama")
    
    tasks = st.session_state.task_manager.tasks
    
    # Filter tasks
    filtered_tasks = [
        t for t in tasks
        if ((show_completed and t.selesai) or (show_incomplete and not t.selesai))
        and (not search_query or search_query.lower() in t.nama.lower())
    ]
    
    if not filtered_tasks:
        st.info("Tidak ada tugas yang ditemukan.")
        return
    
    st.info("💡 Klik pada setiap task untuk melihat detail")
    
    for i, task in enumerate(filtered_tasks, 1):
        status_icon = "✅" if task.selesai else "📌"
        
        with st.expander(f"{i}. {status_icon} {task.nama}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Prioritas:** {task.prioritas}")
                st.write(f"**Deadline:** {task.deadline.strftime('%d %B %Y')}")
                st.write(f"**Estimasi Durasi:** {task.durasi_estimasi:.1f} jam")
                
                if task.selesai:
                    st.write(f"**Status:** Selesai")
                    st.write(f"**Tanggal Selesai:** {task.tanggal_selesai.strftime('%d %B %Y')}")
                    st.write(f"**Durasi Aktual:** {task.durasi_aktual:.1f} jam")
                else:
                    st.write("**Status:** Belum selesai")
                
                if task.waktu_rekomendasi:
                    st.write(f"**Rekomendasi Waktu:** {task.waktu_rekomendasi.strftime('%d %B %Y, %H:%M')}")
            
            with col2:
                st.write(f"**Deskripsi:** {task.deskripsi or 'Tidak ada deskripsi'}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            if not task.selesai:
                if col1.button(f"Tandai Selesai {i}", key=f"complete_{i}"):
                    st.session_state.completing_task = i
                    st.session_state.editing_task = None
                    st.rerun()
            
            if col2.button(f"Edit {i}", key=f"edit_{i}"):
                st.session_state.editing_task = i
                st.session_state.completing_task = None
                st.session_state.edit_form_key = str(uuid.uuid4())
                st.rerun()
            
            if col3.button(f"Hapus {i}", key=f"delete_{i}"):
                st.session_state.task_manager.tasks.remove(task)
                if st.session_state.task_manager.save_to_csv():
                    st.success("Task berhasil dihapus!")
                    st.session_state.editing_task = None
                    st.rerun()
                else:
                    st.error("Gagal menghapus task!")
            
            # Edit form for this task if selected
            if st.session_state.editing_task == i:
                with st.form(key=f"edit_form_{st.session_state.edit_form_key}"):
                    st.subheader(f"Edit Task: {task.nama}")
                    
                    new_nama = st.text_input("Nama Tugas*", value=task.nama)
                    new_deskripsi = st.text_area("Deskripsi Tugas", value=task.deskripsi)
                    new_prioritas = st.selectbox(
                        "Prioritas*", 
                        ["Tinggi", "Sedang", "Rendah"],
                        index=["Tinggi", "Sedang", "Rendah"].index(task.prioritas)
                    )
                    new_deadline = st.date_input(
                        "Deadline*", 
                        value=task.deadline,
                        min_value=date.today()
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        submit_edit = st.form_submit_button("Simpan Perubahan")
                    with col2:
                        cancel_edit = st.form_submit_button("Batal")
                    
                    if submit_edit:
                        if not new_nama:
                            st.error("Nama tugas wajib diisi!")
                        else:
                            task.nama = new_nama
                            task.deskripsi = new_deskripsi
                            task.prioritas = new_prioritas
                            task.deadline = new_deadline
                            
                            st.session_state.task_manager._generate_time_recommendation(task)
                            
                            if st.session_state.task_manager.save_to_csv():
                                st.success("Perubahan berhasil disimpan!")
                                st.session_state.editing_task = None
                                st.rerun()
                            else:
                                st.error("Gagal menyimpan perubahan!")
                    
                    if cancel_edit:
                        st.session_state.editing_task = None
                        st.rerun()
    
    # [Keep your existing completion form code here]
    if st.session_state.completing_task is not None:
        task_index = st.session_state.completing_task - 1
        if task_index < len(filtered_tasks):
            task = filtered_tasks[task_index]
            with st.form(key=f"complete_form_{task_index}"):
                st.write(f"Menandai task '{task.nama}' sebagai selesai")
                tanggal = st.date_input(
                    "Tanggal Selesai", 
                    value=date.today(),
                    key=f"tanggal_{task_index}"
                )
                durasi = st.number_input(
                    "Durasi Aktual (jam)", 
                    min_value=0.1,
                    value=float(task.durasi_estimasi),
                    step=0.5,
                    format="%.1f",
                    key=f"durasi_{task_index}"
                )
                
                submit = st.form_submit_button("Konfirmasi")
                cancel = st.form_submit_button("Batal")
                
                if submit:
                    try:
                        if isinstance(durasi, str):
                            durasi = float(durasi.replace(',', '.'))
                        
                        task.mark_completed(
                            tanggal_selesai=tanggal,
                            durasi_aktual=float(durasi)
                        )
                        if st.session_state.task_manager.save_to_csv():
                            st.success("Task berhasil ditandai selesai!")
                            st.session_state.completing_task = None
                            st.rerun()
                        else:
                            st.error("Gagal menyimpan ke file!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                if cancel:
                    st.session_state.completing_task = None
                    st.rerun()


def show_calendar():
    """Display calendar view of tasks"""
    st.header("🗓 Kalender Tugas")
    
    days = st.slider("Jumlah hari ke depan yang ditampilkan", 1, 30, 7)
    calendar = st.session_state.task_manager.get_tasks_by_deadline(days)
    
    for date in sorted(calendar.keys()):
        with st.expander(f"{date.strftime('%A, %d %B %Y')}"):
            if calendar[date]:
                for task in calendar[date]:
                    status = "✅" if task.selesai else "📌"
                    st.write(f"- {status} {task.nama} ({task.prioritas})")
            else:
                st.write("Tidak ada tugas untuk hari ini.")


def show_statistics():
    """Display basic task statistics"""
    st.header("📊 Statistik")
    
    tab1, tab2 = st.tabs(["Statistik Dasar", "Analisis Lanjutan"])
    
    with tab1:
        st.subheader("Statistik Dasar")
        tasks = st.session_state.task_manager.tasks
        
        if not tasks:
            st.info("Belum ada tugas yang tercatat.")
            return
        
        # Priority distribution
        priority_dist = Counter([t.prioritas for t in tasks])
        st.plotly_chart(
            px.pie(
                names=list(priority_dist.keys()),
                values=list(priority_dist.values()),
                title='Distribusi Prioritas Tugas'
            ),
            use_container_width=True
        )
        
        # Completion rate
        completion_rate = sum(t.selesai for t in tasks) / len(tasks)
        st.metric("Tingkat Penyelesaian", f"{completion_rate*100:.1f}%")
    
    with tab2:
        st.subheader("Analisis Lanjutan")
        tasks = st.session_state.task_manager.tasks
        
        if not tasks:
            st.info("Tidak ada data untuk dianalisis.")
            return
        
        # Deadline analysis
        today = date.today()
        deadline_days = [
            (t.deadline - today).days 
            for t in tasks 
            if not t.selesai and t.deadline >= today
        ]
        
        if deadline_days:
            avg_days = sum(deadline_days)/len(deadline_days)
            st.write(f"⏳ Rata-rata hari menuju deadline: {avg_days:.1f} hari")
            
            st.plotly_chart(
                px.histogram(
                    x=deadline_days,
                    title='Distribusi Hari Menuju Deadline',
                    labels={'x': 'Hari Menuju Deadline'}
                ),
                use_container_width=True
            )


# Recommendations view
def show_recommendations():
    st.header("⏰ Rekomendasi Manajemen Waktu")
    
    active_tasks = st.session_state.task_manager.get_active_tasks()
    
    if not active_tasks:
        st.info("Tidak ada tugas aktif untuk direkomendasikan.")
        return
    
    active_tasks.sort(key=lambda x: (x.prioritas, x.deadline))
    total_time = sum(t.durasi_estimasi for t in active_tasks)
    st.metric("Total Estimasi Waktu untuk Semua Tugas", f"{total_time:.1f} jam")
    
    for i, task in enumerate(active_tasks, 1):
        with st.expander(f"{i}. {task.nama} (Prioritas: {task.prioritas})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Deadline:** {task.deadline.strftime('%A, %d %B %Y')}")
                st.write(f"**Estimasi Durasi:** {task.durasi_estimasi:.1f} jam")
                if task.waktu_rekomendasi:
                    st.write(f"**⏱ Waktu Rekomendasi:** {task.waktu_rekomendasi.strftime('%A, %d %B %Y pukul %H:%M')}")
            
            with col2:
                days_left = (task.deadline - date.today()).days
                urgency = max(0, 10 - days_left) / 10
                st.progress(urgency, text=f"🚦 Tingkat urgensi: {urgency*100:.0f}%")
                
                if days_left < task.durasi_estimasi / 8:
                    st.warning("⚠️ Peringatan: Deadline mungkin tidak tercapai!")


def show_productivity_analysis():
    """Display advanced productivity analysis"""
    st.header("📈 Analisis Produktivitas")
    
    # Debug info
    completed_tasks = [t for t in st.session_state.task_manager.tasks 
                      if t.selesai and t.tanggal_selesai and t.durasi_aktual]
    st.write(f"Jumlah task yang memenuhi syarat: {len(completed_tasks)}")
    
    # Add example data button for testing
    if st.button("Gunakan Data Contoh (Dev Only)"):
        st.session_state.productivity_data = {
            'hourly_productivity': {9: 2.5, 10: 1.8, 14: 3.2},
            'weekday_productivity': {0: 2.1, 1: 1.7, 4: 2.9},
            'productivity_clusters': np.array([[1, 10, 2], [4, 14, 3]]),
            'raw_data': pd.DataFrame()
        }
        st.success("Data contoh berhasil dimuat!")
    
    if st.button("Analisis Pola Produktivitas"):
        with st.spinner("Sedang menganalisis..."):
            try:
                st.session_state.productivity_data = analyze_productivity_patterns(
                    st.session_state.task_manager.tasks
                )
                
                if st.session_state.productivity_data is None:
                    st.warning("Tidak ada data yang bisa dianalisis. Pastikan Anda memiliki:")
                    st.warning("- Minimal 1 tugas yang sudah selesai (✅)")
                    st.warning("- Tugas tersebut memiliki durasi aktual yang tercatat")
                    st.warning("- Tugas memiliki tanggal selesai")
                else:
                    st.success("Analisis berhasil!")
            except Exception as e:
                st.error(f"Terjadi error saat menganalisis: {str(e)}")
                st.error("Pastikan semua task yang selesai memiliki tanggal selesai dan durasi aktual")
    
    if not st.session_state.productivity_data:
        st.info("Klik tombol di atas untuk menganalisis pola produktivitas")
        st.info("Persyaratan data:")
        st.info("- Tugas sudah selesai (✅)")
        st.info("- Memiliki tanggal selesai")
        st.info("- Memiliki durasi aktual")
        return
    
    data = st.session_state.productivity_data
    
    # Show hourly productivity
    st.subheader("Produktivitas per Jam")
    if not data.get('hourly_productivity'):
        st.warning("Tidak ada data produktivitas per jam")
    else:
        hourly_df = pd.DataFrame({
            'Hour': list(data['hourly_productivity'].keys()),
            'Average Duration': list(data['hourly_productivity'].values())
        })
        try:
            st.plotly_chart(
                px.bar(
                    hourly_df,
                    x='Hour',
                    y='Average Duration',
                    title='Rata-rata Durasi Penyelesaian Tugas per Jam'
                ),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Gagal menampilkan grafik: {str(e)}")
    
    # Show weekday productivity
    st.subheader("Produktivitas per Hari dalam Seminggu")
    weekday_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 
                  4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
    
    if not data.get('weekday_productivity'):
        st.warning("Tidak ada data produktivitas per hari")
    else:
        weekday_df = pd.DataFrame({
            'Weekday': [weekday_map[d] for d in data['weekday_productivity'].keys()],
            'Average Duration': list(data['weekday_productivity'].values())
        })
        try:
            st.plotly_chart(
                px.bar(
                    weekday_df,
                    x='Weekday',
                    y='Average Duration',
                    title='Rata-rata Durasi Penyelesaian Tugas per Hari'
                ),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Gagal menampilkan grafik: {str(e)}")
    
    # Show cluster analysis if available
    if data.get('productivity_clusters') is not None:
        st.subheader("Pola Produktivitas (Clustering)")
        st.write("""
            Sistem telah mengidentifikasi pola produktivitas Anda berdasarkan:
            - Hari dalam seminggu
            - Jam produktif
            - Durasi penyelesaian tugas
        """)
        
        try:
            cluster_df = pd.DataFrame(
                data['productivity_clusters'],
                columns=['Hari', 'Jam', 'Durasi']
            )
            cluster_df['Hari'] = cluster_df['Hari'].apply(lambda x: weekday_map[int(round(x))])
            cluster_df['Jam'] = cluster_df['Jam'].apply(lambda x: f"{int(round(x)):02d}:00")
            cluster_df['Durasi'] = cluster_df['Durasi'].round(1)
            
            st.dataframe(cluster_df)
        except Exception as e:
            st.error(f"Gagal menampilkan cluster: {str(e)}")


def main():
    """Main application function"""
    init_session_state()
    setup_page()
    show_main_menu()


if __name__ == "__main__":
    main()