from os import getenv, path

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dotenv import load_dotenv
from flask import send_file, send_from_directory

import utils

load_dotenv()

DATASET_FOLDER = getenv("DATASET_FOLDER")
CSV_FOLDER = getenv("CSV_FOLDER")
CSV_PATH = path.join(CSV_FOLDER,"SpeechBrain", "triplet_df_umap3D.csv")
AUDIO_FOLDER = path.join(DATASET_FOLDER, "fma_large")
VOCAL_AUDIO_FOLDER = path.join(DATASET_FOLDER, "fma_large_stems")
df = pd.read_csv(CSV_PATH)

color_columns = ["genre_top", "creation_date", "release_date", "artist"]
hover_data = ["genre_top", "artist", "album", "track_id"]

audio_choice = ["audio_path", "vocal_audio_path"] # columns containing file paths
is2D = False  # set to False for 3D

def file_name_to_url(file_name):
    if file_name.startswith(VOCAL_AUDIO_FOLDER):
        relative = file_name.replace(VOCAL_AUDIO_FOLDER + "/", "")
        return f"/vocal_audio/{relative}"
    elif file_name.startswith(AUDIO_FOLDER):
        relative = file_name.replace(AUDIO_FOLDER + "/", "")
        return f"/audio/{relative}"
    else:
        # fallback – assume relative path already
        return f"/audio/{file_name}"

# --- Dash app setup ---
app = Dash(__name__)
server = app.server

# Flask routes to serve audio files
@server.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

@server.route("/vocal_audio/<path:filename>")
def serve_vocal_audio(filename):
    return send_from_directory(VOCAL_AUDIO_FOLDER, filename)

customdata_cols = audio_choice + hover_data
def create_figure(color_by, custom_cols):
    if is2D:
        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color=color_by,
            hover_data=hover_data,
            custom_data=custom_cols, 
        )
        fig.update_traces(marker=dict(size=3, opacity=0.9))
        fig.update_layout(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
        )
    else:
        fig = px.scatter_3d(
            df,
            x="UMAP1",
            y="UMAP2",
            z="UMAP3",
            color=color_by,
            hover_data=hover_data,
            custom_data=custom_cols, 
        )
        fig.update_traces(marker=dict(size=3, opacity=0.9))
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, visible=False),
                yaxis=dict(showbackground=False, showgrid=False, visible=False),
                zaxis=dict(showbackground=False, showgrid=False, visible=False),
            )
        )
    return fig

# Initial figure (uses first color column and includes both audio paths in customdata)
fig = create_figure(color_columns[0], customdata_cols)

# Dash Layout
app.layout = html.Div(
    [
        dcc.Dropdown(
            id="color-dropdown",
            options=[{"label": col, "value": col} for col in color_columns],
            value=color_columns[0] if color_columns else None,
            clearable=False,
            style={"width": "500px", "marginBottom": "20px"},
        ),
        dcc.Dropdown(
            id="audio-source",
            options=[{"label": col, "value": col} for col in audio_choice],
            value=audio_choice[0],
            clearable=False,
            style={"width": "500px", "marginBottom": "20px"},
        ),
        dcc.Graph(
            id="voice-plot",
            figure=fig,
            style={"height": "80vh"},
        ),
        html.Div(id='current-song-info', style={'marginBottom': '10px', 'fontWeight': 'bold'}),
        html.Audio(
            id="audio-player",
            src="",
            controls=True,
            autoPlay=True,
            style={"width": "100%", "marginTop": "20px"},
        ),
    ]
)

# Callback: update plot color by dropdown selection
@app.callback(
    Output("voice-plot", "figure"),
    Input("color-dropdown", "value"),
)
def update_graph(selected_color):
    return create_figure(selected_color, audio_choice)

@app.callback(
    [Output("audio-player", "src"),
     Output("current-song-info", "children")],
    [Input("voice-plot", "clickData"),
     Input("audio-source", "value")],
    prevent_initial_call=True
)
def play_audio_and_show_info(clickData, selected_audio_col):
    if clickData and "points" in clickData and clickData["points"]:
        pt = clickData["points"][0]
        col_index = audio_choice.index(selected_audio_col)
        file_path = pt["customdata"][col_index]
        # Extract metadata (indices after audio_choice)
        genre = pt["customdata"][2]   # genre_top
        artist = pt["customdata"][3]  # artist
        album = pt["customdata"][4]   # album
        info = f"{artist} – {album} ({genre})"
        return file_name_to_url(file_path), info
    return "", ""

# Main loop
if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")