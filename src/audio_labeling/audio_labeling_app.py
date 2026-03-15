import os
import tkinter as tk
from tkinter import ttk

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..globals import CSV_FOLDER
from .dataset_handler import DatasetHandler

# ---------------------- Consts  ----------------------

# Initial path: dataset_path=os.path.join(CSV_FOLDER, "LargeDataset", "dataset.csv")
dh = DatasetHandler()

# Spectrogram Settings
HOP_LENGTH = 2048
MAX_FREQ = 8_000

# ---------------------- Methods  ----------------------

def update_plot():
    ax.clear()
    y = dh.current_row["y"]
    sr = dh.current_row["sr"]
    stft = librosa.stft(y, hop_length=HOP_LENGTH)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    librosa.display.specshow(db, sr=sr, hop_length=HOP_LENGTH, 
                             y_axis='hz', ax=ax, y_coords=None)
    ax.set_ylim(0, MAX_FREQ)
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)")
    canvas.draw()

def update_display():
    """Refresh all display widgets with data from dh.current_row."""
    track_id_label.config(text=f"Track ID: {dh.current_row['track_id']}")
    checked_label.config(text=f"Checked: {dh.current_row['checked']}")
    is_voiced_label.config(text=f"Is Voiced: {dh.current_row['is_voiced']}")
    vocal_length_label.config(text=f"Vocal Length (s): {dh.current_row['vocal_content_length_s']:.2f}")
    
    # Progress info
    progress = dh.get_progress()
    total_label.config(text=f"Total: {progress["total"]}")
    checked_count_label.config(text=f"Checked: {progress["checked"]}")
    progress_label.config(text=f"Progress: {progress["percent"]:.1f}%")
    
    voice_quality_var.set(int(dh.current_row.get("voice_quality", 0)))
    multiple_voices_var.set(bool(dh.current_row.get("multiple_voices", False)))
    interview_var.set(bool(dh.current_row.get("interview", False))) 

    dh.play_audio()
    update_plot()

def on_voice_quality_change(*args):
    """Update dh.current_row when slider moves."""
    dh.current_row["voice_quality"] = voice_quality_var.get()

def on_multiple_voices_change():
    """Update dh.current_row when checkbox toggles."""
    dh.current_row["multiple_voices"] = multiple_voices_var.get()

def on_interview_change():
    """Update dh.current_row when interview checkbox toggles."""
    
    dh.current_row["interview"] = interview_var.get()

def submit_and_next(event=None):
    """Call setAndMoveOn and update display."""
    dh.set_row()
    update_display()

def go_forward():
    dh.navigate(1)
    update_display()

def go_backward():
    dh.navigate(-1)
    update_display()

def on_closing():
    """Save dataset when window is closed."""
    dh.save()
    canvas.get_tk_widget().destroy()
    plt.close(fig)
    root.destroy()

# ---------------------- GUI Setup ----------------------
root = tk.Tk()
root.title("Audio Labeling Tool")
root.geometry("600x700")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Read-only info frame
info_frame = ttk.LabelFrame(root, text="Track Info", padding=10)
info_frame.pack(fill="x", padx=10, pady=5)

track_id_label = ttk.Label(info_frame,     text="Track ID:         ")
track_id_label.pack(anchor="w")
checked_label = ttk.Label(info_frame,      text="Checked:          ")
checked_label.pack(anchor="w")
is_voiced_label = ttk.Label(info_frame,    text="Is Voiced:        ")
is_voiced_label.pack(anchor="w")
vocal_length_label = ttk.Label(info_frame, text="Vocal Length (s): ")
vocal_length_label.pack(anchor="w")

total_label = ttk.Label(info_frame,         text="Total:    ")
total_label.pack(anchor="w")
checked_count_label = ttk.Label(info_frame, text="Checked:  ")
checked_count_label.pack(anchor="w")
progress_label = ttk.Label(info_frame,      text="Progress: ")
progress_label.pack(anchor="w")

# Input frame
input_frame = ttk.LabelFrame(root, text="Labeling", padding=10)
input_frame.pack(fill="x", padx=10, pady=5)

# Slider for voice_quality (0-3)
ttk.Label(input_frame, text="Voice Quality (0-3):").pack(anchor="w")
voice_quality_var = tk.IntVar(value=0)
voice_quality_var.trace_add("write", on_voice_quality_change)
voice_quality_spinbox = tk.Spinbox(
    input_frame, from_=0, to=3, textvariable=voice_quality_var,
    width=10, command=on_voice_quality_change
)
voice_quality_spinbox.pack(anchor="w", pady=5)

# Checkbox for multiple_voices
multiple_voices_var = tk.BooleanVar(value=False)
#multiple_voices_var.trace('w', lambda *args: on_multiple_voices_change())
multiple_voices_check = ttk.Checkbutton(
    input_frame, text="Multiple Voices", variable=multiple_voices_var,
    command=on_multiple_voices_change,
    onvalue=True, offvalue=False
)
multiple_voices_check.pack(anchor="w", pady=5)
interview_var = tk.BooleanVar(value=False)
#interview_var.trace('w', lambda *args: on_interview_change())
interview_check = ttk.Checkbutton(
    input_frame, text="Interview", variable=interview_var,
    command=on_interview_change,
    onvalue=True, offvalue=False
)
interview_check.pack(anchor="w", pady=5)

# Audio control frame
audio_frame = ttk.LabelFrame(root, text="Audio Playback", padding=10)
audio_frame.pack(fill="x", padx=10, pady=5)

play_button = ttk.Button(audio_frame, text="Play", command=dh.play_audio)
play_button.pack()

plot_frame = ttk.LabelFrame(root, text="Waveform", padding=10)
plot_frame.pack(fill="both", expand=True, padx=10, pady=5)
fig, ax = plt.subplots(figsize=(5, 2), dpi=80)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)


# Navigation frame
nav_frame = ttk.Frame(root, padding=10)
nav_frame.pack(fill="x", padx=10, pady=5)

# Center frame for buttons
button_frame = ttk.Frame(nav_frame)
button_frame.pack(expand=True)

back_button = ttk.Button(button_frame, text="< Back", command=go_backward)
back_button.pack(side="left", padx=5)
submit_button = ttk.Button(button_frame, text="Submit", command=submit_and_next)
submit_button.pack(side="left", padx=5)
forward_button = ttk.Button(button_frame, text="Forward >", command=go_forward)
forward_button.pack(side="left", padx=5)

# -- Keybindings --
def on_up_arrow(event):
    current_val = voice_quality_var.get()
    if current_val < 3:
        voice_quality_var.set(current_val + 1)

def on_down_arrow(event):
    current_val = voice_quality_var.get()
    if current_val > 0:
        voice_quality_var.set(current_val - 1)

def on_m_key(event):
    multiple_voices_var.set(not multiple_voices_var.get())
    on_multiple_voices_change()

def on_i_key(event):
    interview_var.set(not interview_var.get())
    on_interview_change()

root.bind('<Up>', on_up_arrow)
root.bind('<Down>', on_down_arrow)
root.bind('<Return>', submit_and_next)
root.bind('<space>', lambda event: dh.play_audio())
root.bind('<Left>', lambda event: go_backward())
root.bind('<Right>', lambda event: go_forward())
root.bind('m', on_m_key)
root.bind('M', on_m_key)
root.bind('i', on_i_key)
root.bind('I', on_i_key)


# -- Start the Main Loop --
update_display()
root.mainloop()