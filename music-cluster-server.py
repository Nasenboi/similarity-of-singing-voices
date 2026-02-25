from os import getenv, path

import dash
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dotenv import load_dotenv
from flask import send_file, send_from_directory

import utils

load_dotenv()


DATASET_FOLDER = getenv("DATASET_FOLDER")
METADATA_FOLDER = path.join(DATASET_FOLDER, "fma_metadata")
AUDIO_FOLDER = path.join(DATASET_FOLDER, "fma_large")
CSV_PATH = path.join(METADATA_FOLDER, "tracks_df_embedding_3D.csv")

df = utils.load(CSV_PATH)
new_columns = list(df.columns)
new_columns[-4:] = [('track', 'audio_path'), ('UMAP', 'X'), ('UMAP', 'Y'), ('UMAP', 'Z')]
df.columns = pd.MultiIndex.from_tuples(new_columns)
df.columns = ['_'.join(col).rstrip('_') for col in df.columns]

df["track_audio_path"] = [
    utils.get_audio_path(AUDIO_FOLDER, i) for i in df.index
]

color_columns = ["track_genre_top", "album_date_created", "album_date_released", "artist_location"]
hover_data = ["track_genre_top", "artist_name", "album_title"]

def file_name_to_url(file_name):
    file_name = file_name.replace(AUDIO_FOLDER+"/", "")
    return f"/audio/{file_name}"



# --- Dash app setup ---
app = Dash(__name__)
server = app.server


# Flask route to serve audio files
@server.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)


# Initial 3D scatter plot with hover info
fig = px.scatter_3d(
    df,
    x="UMAP_X",
    y="UMAP_Y", 
    z="UMAP_Z",
    color=color_columns[0],
    hover_data=hover_data,
    custom_data=["track_audio_path"],
)
fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False, showgrid=False, visible=False),
        yaxis=dict(showbackground=False, showgrid=False, visible=False),
        zaxis=dict(showbackground=False, showgrid=False, visible=False)
    )
)

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
        dcc.Graph(
            id="voice-3d-plot",
            figure=fig,
            style={"height": "80vh"},
        ),
        html.Audio(
            id="audio-player",
            src="",
            controls=True,
            autoPlay=True,
            style={"width": "100%", "marginTop": "20px"},
        ),
    ]
)


# Callback: play audio when point is clicked
@app.callback(
    Output("audio-player", "src"),
    Input("voice-3d-plot", "clickData"),
    prevent_initial_call=True,
)
def play_audio(clickData):
    if clickData and "points" in clickData and clickData["points"]:
        file_name = clickData["points"][0]["customdata"][0]
        return file_name_to_url(file_name)
    return ""


# Callback: update 3D plot color by dropdown selection
@app.callback(
    Output("voice-3d-plot", "figure"),
    Input("color-dropdown", "value"),
)
def updateGraph(selected_color):
    fig = px.scatter_3d(
        df,
        x="UMAP_X",
        y="UMAP_Y", 
        z="UMAP_Z",
        color=selected_color,
        hover_data=hover_data,
        custom_data=["track_audio_path"],
    )
    fig.update_traces(marker=dict(size=3, opacity=0.9)) 
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, visible=False)
        )
    )
    return fig

# Main loop
if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")