from dash import Dash, html, callback, Input, Output, State, dcc, dash_table, no_update
import dash_cytoscape as cyto
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os
import numpy as np
import colorlover
from tqdm.notebook import tqdm
from scipy.spatial.distance import euclidean
from math import acos, sqrt, atan2, pi
from sklearn.preprocessing import LabelEncoder

colors = {"background": "#111111", "text": "#7FDBFF"}
app = Dash(__name__)

server = app.server

# Define data paths
data_dir = Path("/Users/keltim01/Documents/data/nfl-big-data-bowl-2023")
model_dir = data_dir.joinpath("model_data")

df_players = pd.read_csv(data_dir.joinpath("players.csv"))
df_plays = pd.read_csv(data_dir.joinpath("plays.csv"))
df_games = pd.read_csv(data_dir.joinpath("games.csv"))
df_pff_data = pd.read_csv(data_dir.joinpath("pffScoutingData.csv"))
df_nfl_pbp = pd.read_parquet(data_dir.joinpath("play_by_play_2021.parquet"))
df_weekly_data = pd.read_parquet(data_dir.joinpath("standardized","all_weeks.parquet"))

# define colors
color_football = ["#CBB67C", "#663831"]

# Importing the Team color information from nflfastr (https://www.nflfastr.com/reference/teams_colors_logos.html?q=colors#null)
df_teams = pd.read_csv(data_dir.joinpath("teams.csv"))

# define scale for the field
scale = 12

# Feature Engineering Functions
# Define utility functions
def euclidean_distance(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_angle(x1, y1, x2, y2):
    dot_product = x1 * x2 + y1 * y2
    magnitude1 = sqrt(x1**2 + y1**2)
    magnitude2 = sqrt(x2**2 + y2**2)
    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle = acos(np.clip(cos_angle, -1, 1)) * (180 / pi)  # Clamp value and convert to degrees
    return angle

def cosine_similarity(x1, y1, x2, y2):
    dot_product = x1 * x2 + y1 * y2
    magnitude1 = sqrt(x1**2 + y1**2)
    magnitude2 = sqrt(x2**2 + y2**2)
    return dot_product / (magnitude1 * magnitude2)

def orthogonal_distance(x1, y1, x2, y2, x3, y3):
    num = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)
    den = sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / den

def calculate_sideline_distance(y_position, sideline1=0, sideline2=53.3):
    distance_to_sideline1 = abs(y_position - sideline1)
    distance_to_sideline2 = abs(y_position - sideline2)
    return min(distance_to_sideline1, distance_to_sideline2)


def process_data_pressure_probability(
        df_weekly_data: pd.DataFrame,
        df_pff_data: pd.DataFrame,
        df_plays: pd.DataFrame,
):
    # Mutate and calculate second_since_snap
    df_weekly_data['snap_time'] = df_weekly_data.loc[df_weekly_data['event'] == 'ball_snap', 'second'].groupby([df_weekly_data['gameId'], df_weekly_data['playId']]).transform('first')
    df_weekly_data['second_since_snap'] = df_weekly_data['second'] - df_weekly_data['snap_time']
    df_weekly_data = df_weekly_data.drop(columns=['snap_time'])

    # Selecting columns from pffData
    df_pff_join = df_pff_data[['gameId', 'playId', 'nflId', 'pff_role', 'pff_hit', 'pff_hurry', 'pff_sack',
                        'pff_beatenByDefender', 'pff_hurryAllowed', 'pff_hitAllowed', 'pff_sackAllowed',
                        'pff_nflIdBlockedPlayer', 'pff_blockType']].copy()
    
    # Join and create pressure_df
    df_pressure = pd.merge(df_weekly_data, df_pff_join, on=['gameId', 'playId', 'nflId'], how='left').fillna(0)
    df_pressure['is_pressure'] = np.where((df_pressure['pff_hit'] + df_pressure['pff_hurry'] + df_pressure['pff_sack']) > 0, 1, 0)


    # Create rusher_df
    df_rusher = df_pressure[df_pressure['pff_role'] == 'Pass Rush'].drop(columns=['pff_hurryAllowed', 'pff_sackAllowed',
                                                                                'pff_hitAllowed', 'pff_beatenByDefender',
                                                                                'pff_blockType', 'pff_nflIdBlockedPlayer'])
    
    df_rush_join = df_rusher[['gameId', 'playId', 'frameId', 'nflId', 'x', 'y']].copy() 

    # Create qb_df
    df_qb = (df_pressure[df_pressure['pff_role'] == 'Pass']
            .rename(columns={col: f'qb_{col}' for col in ['x', 'y', 's', 'a', 'dis', 'o', 'dir']})
            .rename(columns={'nflId': 'qb_nflId'}))
    
    df_qb_join = (df_qb[['gameId', 'playId', 'frameId', 'qb_nflId', 'qb_x', 'qb_y', 'qb_s', 'qb_a', 'qb_o', 'qb_dir', 'qb_dis']].copy())
    
    # Create blocker_df
    df_blocker = (df_pressure[df_pressure['pff_role'] == 'Pass Block']
                .rename(columns={col: f'blocker_{col}' for col in ['x', 'y', 's', 'a', 'dis', 'o', 'dir']})
                .drop(columns=['pff_hit', 'pff_hurry', 'pff_sack', 'time']))

    df_blk_join = (df_blocker[['gameId', 'playId', 'frameId', 'nflId', 'blocker_x', 'blocker_y', 'blocker_s',
                            'blocker_a', 'blocker_o', 'blocker_dir', 'blocker_dis', 'pff_role',
                            'pff_beatenByDefender', 'pff_hurryAllowed', 'pff_hitAllowed',
                            'pff_sackAllowed', 'pff_blockType']].copy()
                .rename(columns={'nflId': 'blocker_id', 'pff_role': 'blocker_role'}))
    
    # Cross join and calculate distances
    df_cross_join = pd.merge(df_rush_join, df_blk_join, on=['gameId', 'playId', 'frameId'], how='left')
    df_cross_join['blk_rush_dist'] = df_cross_join.apply(lambda row: euclidean_distance(row['x'], row['y'], row['blocker_x'], row['blocker_y']), axis=1)


    # Find top 3 shortest distances
    df_blocker_top_3 = (df_cross_join.sort_values(by='blk_rush_dist')
                    .groupby(['gameId', 'playId', 'frameId', 'nflId'])
                    .head(3)
                    .assign(player_dist_rank=lambda x: x.groupby(['gameId', 'playId', 'frameId', 'nflId']).cumcount() + 1))


    # Merge with qb_df and rusher_df
    df_pres_long = (df_blocker_top_3.merge(df_qb_join, on=['gameId', 'playId', 'frameId'], how='left')
                    .merge(df_rusher, on=['gameId', 'playId', 'frameId', 'nflId', 'x', 'y'], how='left'))
    
        # Calculate angles and other features
    df_with_angles = df_pres_long.assign(
        vec_rusher_to_qb_x=lambda x: x['qb_x'] - x['x'],
        vec_rusher_to_qb_y=lambda x: x['qb_y'] - x['y'],
        vec_rusher_to_blocker_x=lambda x: x['blocker_x'] - x['x'],
        vec_rusher_to_blocker_y=lambda x: x['blocker_y'] - x['y'],
        leverage_angle=lambda x: x.apply(lambda row: calculate_angle(row['vec_rusher_to_qb_x'], row['vec_rusher_to_qb_y'], row['vec_rusher_to_blocker_x'], row['vec_rusher_to_blocker_y']), axis=1),
        cos_sim=lambda x: x.apply(lambda row: cosine_similarity(row['vec_rusher_to_qb_x'], row['vec_rusher_to_qb_y'], row['vec_rusher_to_blocker_x'], row['vec_rusher_to_blocker_y']), axis=1),
        ortho_dist=lambda x: x.apply(lambda row: orthogonal_distance(row['x'], row['y'], row['qb_x'], row['qb_y'], row['blocker_x'], row['blocker_y']), axis=1),
        blocker_influence=lambda x: x.apply(lambda row: np.where(row['ortho_dist'] <= 1, 1, np.exp(-row['ortho_dist'])), axis=1),
        rel_s=lambda x: x['s'] - x['qb_s'],
        approach_angle=lambda x: x.apply(lambda row: np.arctan2(row['y'] - row['qb_y'], row['x'] - row['qb_x']), axis=1),
        rel_o=lambda x: np.abs(x['o'] - x['qb_o']),
        qb_dist_near_sideline=lambda x: x['qb_y'].apply(calculate_sideline_distance),
        rush_qb_dist=lambda x: x.apply(lambda row: euclidean_distance(row['x'], row['y'], row['qb_x'], row['qb_y']), axis=1)
        )

    # Summarize blocker-interference
    df_blocker_interference = df_with_angles.groupby(['gameId', 'playId', 'frameId', 'nflId']).agg(blocker_interference=('blocker_influence', 'sum')).reset_index()

    # Pivot the dataframe to wide format
    df_wide = df_with_angles.pivot(index=['gameId', 'playId', 'frameId', 'nflId'],
                                columns='player_dist_rank',
                                values=['blocker_id', 'blocker_x', 'blocker_y', 'blocker_s', 'blocker_a',
                                        'blocker_o', 'blocker_dir', 'blocker_dis', 'pff_blockType',
                                        'pff_beatenByDefender', 'pff_hurryAllowed', 'pff_hitAllowed',
                                        'pff_sackAllowed', 'blk_rush_dist', 'vec_rusher_to_blocker_x',
                                        'vec_rusher_to_blocker_y', 'leverage_angle', 'cos_sim', 'ortho_dist',
                                        'blocker_influence'])

    df_wide.columns = ['{}_{}'.format(col[0], col[1]) for col in df_wide.columns]
    df_wide.reset_index(inplace=True)

    # Join with play features
    play_feats = df_plays[['gameId', 'playId', 'down', 'yardsToGo', 'defendersInBox', 'absoluteYardlineNumber']]
    df_pres_model = pd.merge(df_wide, play_feats, on=['gameId', 'playId'], how='left')

    return df_pres_model



def plot_field(line_of_scrimmage, first_down_line):
    field_data = []

    field_data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )
    field_data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[53.5 - 5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )

    # plot line of scrimmage
    field_data.append(
        go.Scatter(
            x=[line_of_scrimmage, line_of_scrimmage],
            y=[0, 53.5],
            line_dash="dash",
            line_color="blue",
            showlegend=False,
            hoverinfo="none",
        )
    )

    # plot first down line
    field_data.append(
        go.Scatter(
            x=[first_down_line, first_down_line],
            y=[0, 53.5],
            line_dash="dash",
            line_color="red",
            showlegend=False,
            hoverinfo="none",
        )
    )

    return field_data


# Create the Elements dictionary for the Dash Cytoscape Graph
def frame_data(df_frame, ball_carrier_id, tackle_id, n_clicks=0):
    # df_frame.loc[df_frame["nflId"] == ball_carrier_id, "club"] = "bc"
    # df_frame.loc[df_frame["nflId"].isin(tackle_id), "club"] = "tackle"
    data = (
        df_frame[["nflId", "displayName", "jerseyNumber", "x", "y", "club"]]
        .apply(
            lambda x: {
                "data": {
                    "id": "{}-{}".format(x["nflId"], n_clicks),
                    "label": x["jerseyNumber"],
                    "classes": x["club"],
                },
                "position": {"x": (x["x"]) * scale, "y": (53.3 - x["y"]) * scale},
                "grabbable": False if x["club"] == "football" else True,
            },
            axis=1,
        )
        .tolist()
    )
    return data

def discrete_background_color_bins(df, n_bins=5, columns='all'):
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in df:
            df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df.select_dtypes('number')
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Greens'][i - 1]
        color = 'white' if i > len(bounds) / 2. else 'inherit'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
    return styles


# Create the data frame for the adjusted coordinates from the Dash Cytoscape Graph
def create_data(elements, df_frame, scale=15):
    df = pd.json_normalize(elements)
    df.rename(
        columns={
            "data.id": "nflId",
            "data.label": "jerseyNumber",
            "data.classes": "club",
            "position.x": "x",
            "position.y": "y",
        },
        inplace=True,
    )
    df["nflId"] = df["nflId"].str.split("-").str[0].astype("float64")
    # df['x'] = df['x'].apply(lambda x: 120 - (x / scale))
    # df['y'] = df['y'].apply(lambda x: x / scale)
    df["x"] = df["x"].apply(lambda x: round((x / scale), 2))
    df["y"] = df["y"].apply(lambda x: round(53.3 - (x / scale), 2))

    # TODO: Calculate new values for s, a, dis, o, dir
    old_columns = [
        "gameId",
        "playId",
        "nflId",
        "displayName",
        "frameId",
        "time",
        "jerseyNumber",
        "club",
        "playDirection",
        "s",
        "a",
        "dis",
        "o",
        "dir",
        "event",
        "second",
    ]
    new_colums = ["nflId", "x", "y"]
    df = pd.merge(df_frame[old_columns], df[new_colums], on="nflId", how="left")

    return df


# Getting all the information together for the application
# Select the week 3
gameId = 2021091200

df_game_plays = df_plays[df_plays.gameId == gameId]
initial_playId = df_game_plays["playId"].sort_values().unique()[0]

df_initial_play = df_plays[(df_plays.gameId == gameId) & (df_plays.playId == initial_playId)]

df_tracking_initial_play = df_weekly_data[
    (df_weekly_data.gameId == gameId) & (df_weekly_data.playId == initial_playId)
]
# df_tracking_initial_play = df_model_results[(df_model_results.playId == initial_playId)]
initial_frames = df_tracking_initial_play["frameId"].unique()
initial_frameId = df_tracking_initial_play["frameId"].unique()[0]

# df_initial_frame = weekly_data[
#     (weekly_data["gameId"] == gameId)
#     & (weekly_data["playId"] == initial_playId)
#     & (weekly_data["frameId"] == initial_frameId)
# ]

df_initial_frame = df_tracking_initial_play[df_tracking_initial_play["frameId"] == initial_frameId]

# df_model_results_frame = df_model_results[
#     (df_model_results["gameId"] == gameId)
#     & (df_model_results["playId"] == initial_playId)
#     & (df_model_results["frameId"] == initial_frameId)
# ].copy()
# df_model_results_frame["tackle_prob_new"] = df_model_results_frame["tackle_prob"]
# df_model_results_frame["tackle_prob_delta"] = (
#     df_model_results_frame["tackle_prob_new"] - df_model_results_frame["tackle_prob"]
# )
# df_model_results_frame["xYAC_new"] = df_model_results_frame["xYAC"]
# df_model_results_frame["xYAC_delta"] = (
#     df_model_results_frame["xYAC_new"] - df_model_results_frame["xYAC"]
# )

# df_model_results_frame = df_model_results_frame[
#     df_model_results_frame["club"] != "football"
# ]

# df_results_bc = df_model_results_frame.loc[
#     df_model_results_frame["isBallCarrier"] == 1,
#     ["displayName", "jerseyNumber", "club", "xYAC", "xYAC_new", "xYAC_delta"],
# ].copy()
# df_results_bc = df_results_bc.round(2)
# df_results_bc.rename(
#         columns = {
#             "displayName": "Player",
#             "jerseyNumber": "Jersey Number",
#             "club": "Team",
#             "xYAC": "Expected Yards",
#             "xYAC_new": "New Expected Yards",
#             "xYAC_delta": "Delta Expected Yards",
#         },
#     )

# df_results_tackle = df_model_results_frame.loc[
#     ~df_model_results_frame["tackle_prob"].isna(),
#     [
#         "displayName",
#         "jerseyNumber",
#         "club",
#         "is_tackle",
#         "tackle_prob",
#         "tackle_prob_new",
#         "tackle_prob_delta",
#     ],
# ].copy()
# df_results_tackle = df_results_tackle.round(2)
# df_results_tackle.rename(
#     columns = {
#         "displayName": "Player",
#         "jerseyNumber": "Jersey Number",
#         "club": "Team",
#         "is_tackle": "Tackle",
#         "tackle_prob": "Tackle Probability",
#         "tackle_prob_new": "New Tackle Probability",
#         "tackle_prob_delta": "Delta Tackle Probability",
#     }
# )

# line_of_scrimmage = 110 - df_initial_play.absoluteYardlineNumber.values[0]
# first_down_line =110 - (
#     df_initial_play.absoluteYardlineNumber.values[0]
#     + df_initial_play.yardsToGo.values[0]
# )

if df_tracking_initial_play.playDirection.values[0] == "right":
    line_of_scrimmage = df_initial_play.absoluteYardlineNumber.values[0]
    first_down_line = (
        df_initial_play.absoluteYardlineNumber.values[0] + df_initial_play.yardsToGo.values[0]
    )
else:
    line_of_scrimmage = 120 - df_initial_play.absoluteYardlineNumber.values[0]
    first_down_line = (
        120 - df_initial_play.absoluteYardlineNumber.values[0] + df_initial_play.yardsToGo.values[0]
    )

teams = df_game_plays["possessionTeam"].unique()

# ball_carrier_id = df_model_results_frame["ballCarrierId"].values[0]
# tackle_id = df_model_results_frame.loc[df_model_results_frame['is_tackle'] == 1, 'nflId'].values
# data = frame_data(df_frame=df_initial_frame, tackle_id=tackle_id, ball_carrier_id=ball_carrier_id)
data = frame_data(df_frame=df_initial_frame, tackle_id=None, ball_carrier_id=None)

# Create interactice figure
layout_interactive = go.Layout(
    autosize=True,
    width=120 * scale,
    height=53.3 * scale,
    xaxis=dict(
        range=[0, 120],
        autorange=False,
        tickmode="array",
        tickvals=np.arange(10, 111, 5).tolist(),
        showticklabels=False,
    ),
    yaxis=dict(range=[0, 53.3], autorange=False, showgrid=False, showticklabels=False),
    plot_bgcolor="#00B140",
    margin=dict(l=0, r=0, t=0, b=0),
)


fig_interactive = go.Figure(
    data=plot_field(
        line_of_scrimmage=line_of_scrimmage, first_down_line=first_down_line
    ),
    layout=layout_interactive,
)

stylesheet = [
    {"selector": "node", "style": {"content": "data(label)"}},
    {
        "selector": '[classes = "football"]',
        "style": {
            "width": 15,
            "height": 10,
            "background-color": color_football[0],
            "border-color": color_football[1],
            "border-width": 2,
        },
    },
    {
        "selector": '[classes = "{}"]'.format(teams[0]),
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": df_teams.loc[
                df_teams["team_abbr"] == teams[0], "team_color"
            ].values[0],
            "border-color": df_teams.loc[
                df_teams["team_abbr"] == teams[0], "team_color2"
            ].values[0],
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
    {
        "selector": '[classes = "{}"]'.format(teams[1]),
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": df_teams.loc[
                df_teams["team_abbr"] == teams[1], "team_color"
            ].values[0],
            "border-color": df_teams.loc[
                df_teams["team_abbr"] == teams[1], "team_color2"
            ].values[0],
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
    {
        "selector": '[classes = "bc"]',
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": "red",
            "border-color": "white",
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
    {
        "selector": '[classes = "bc"]',
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": "red",
            "border-color": "white",
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
    {
        "selector": '[classes = "tackle"]',
        "style": {
            "width": 22.5,
            "height": 22.5,
            "background-color": "#ffcc00",
            "border-color": "white",
            "border-width": 2.5,
            "font-size": 12.5,
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
        },
    },
]

# Define the layout of the dashboard
app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Markdown('''
                    **How to use the app:**
                    * Select the play you want to look at from the dropdown menu.
                    * Use the slider to move through the frames of the play.
                    * Drag the players to the positions where you want them.
                        * The red dot with the white border is the ball carrier.
                        * The yellow dot with the white border is the tackler.
                        * The orange dots with the black border are Cincinnati Bengals players.
                        * The teal dots with the orange border are Miami Dolphins players.
                    * **Score Modified Play** - Click this button to recalculate the **Expected Yards** and **Tackle Probability** metrics for the new positions.
                    * **Start/Stop Play** - Click this button to start and stop the animiation of the play in a loop (The animation also stops by selecting a new play).
                    * In the tables down below the results for the Metrics Expected Yards and Tackle Probability are shown for the Ball Carrier and the Possible Tacklers.
                        * **Expected Yards** - The expected yards for the ball carrier on the play. This is calculated using the xYAC model.
                        * **Tackle Probability** - The probability that the player will make the tackle on the ball carrier. This is calculated using the pursuit model.
                    * After scoring the modified play, the table shows the results for the ball carrier and the possible tacklers for the modified play and the difference from the original play.
                    
                    Note: 
                    * The plays start with the second available frame. 
                    * For run plays the Expected Yards metric is available from the moment the ball is snapped.
                ''')
            ]
        ),
        html.Br(),
        html.Div(
            [
                dcc.Dropdown(
                    options=df_game_plays[["playId", "playDescription"]]
                    .sort_values("playId")
                    .apply(
                        lambda x: {
                            "label": "{} - {}".format(
                                x["playId"], x["playDescription"]
                            ),
                            "value": x["playId"],
                        },
                        axis=1,
                    )
                    .values.tolist(),
                    value=df_game_plays["playId"].sort_values().unique()[0],
                    clearable=False,
                    id="play-selection",
                ),
                html.Br(),
                html.Div(
                    [
                        html.Button("Start/Stop Play", id="btn-play", style={"margin-left": "10px"}),
                    ]
                ),
                html.Br(),
                dcc.Slider(
                    min=2,
                    max=len(initial_frames),
                    step=1,
                    value=2,
                    marks=dict(
                        zip(
                            [x for x in range(2, len(initial_frames) + 1)],
                            [{"label": "{}".format(int(x))} for x in initial_frames[1:]],
                        )
                    ),
                    id="slider-frame",
                ),
            ],
            style={"width": "75%", "display": "inline-block"},
        ),
        html.Br(),
        html.Img(
            id='home-logo',
            src=df_teams[df_teams['team_abbr'] == 'CIN']['team_logo_wikipedia'].values[0],
            height=25,
            style={"margin-left": "10%"}
        ),
        html.H1(
            id="score",
            children="""{} - {}""".format(df_initial_play['preSnapHomeScore'].values[0], df_initial_play['preSnapVisitorScore'].values[0]),
            style={"float": "center", "display": "inline-block", "margin-left": "10px"},
        ),
        html.Img(
            id='away-logo',
            src=df_teams[df_teams['team_abbr'] == 'MIA']['team_logo_wikipedia'].values[0],
            height=25,
            style={"margin-left": "10px"}
        ),
        html.H1(
            id="down-distance",
            children="""{} & {}""".format(df_initial_play['down'].values[0], df_initial_play['yardsToGo'].values[0]),
            style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        ),
        html.H1(
            id="time",
            children="""{}""".format(df_initial_play['gameClock'].values[0]),
            style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        ),
        html.H1(
            id="quarter",
            # children="""{}th Quarter""".format(df_initial_play['quarter'].values[0]),
            children='4th Quarter' if df_initial_play['quarter'].values[0] == 4 else '1st Quarter' if df_initial_play['quarter'].values[0] == 1 else '2nd Quarter' if df_initial_play['quarter'].values[0] == 2 else '3rd Quarter',
            style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        ),
        html.H1(
            id="possession-name",
            children="""Possession Team: """,
            style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        ),
        html.Img(
            id='possession',
            src=df_teams[df_teams['team_abbr'] == df_initial_play['possessionTeam'].values[0]]['team_logo_wikipedia'].values[0],
            height=25,
            style={"margin-left": "10px"}
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="plot-field",
                            figure=fig_interactive,
                        ),
                    ],
                    style={
                        "position": "absolute",
                        "width": "{}px".format(120 * scale),
                        "height": "{}px".format(53.3 * scale),
                    },
                ),
                html.Div(
                    [
                        cyto.Cytoscape(
                            id="cytoscape-test",
                            layout={
                                "name": "preset",
                                "fit": True,
                            },
                            style={
                                "position": "absolute",
                                "width": "{}px".format(120 * scale),
                                "height": "{}px".format(53.3 * scale),
                            },
                            elements=data,
                            zoom=1,
                            zoomingEnabled=False,
                            panningEnabled=False,
                            stylesheet=stylesheet,
                        ),
                    ]
                ),
            ]
        ),
        
        html.Div([html.Br() for x in range(36)]),
        html.Br(),
        html.Div(
            [
                html.Button("Score Modified Play", id="btn-click"),
            ]
        ),
        # html.Div(
        #     [
        #         html.H3("Ball Carrier Results"),
        #         dash_table.DataTable(
        #             data=df_results_bc.to_dict("records"),
        #             columns=[{"name": i, "id": i} for i in df_results_bc.columns],
        #             style_data_conditional=discrete_background_color_bins(df_results_bc, columns=['xYAC', 'xYAC_new', 'xYAC_delta']),
        #             id="tbl-results-bc",
        #         ),
        #         html.H3("Possible Tackler Results"),
        #         dash_table.DataTable(
        #             data=df_results_tackle.to_dict("records"),
        #             columns=[{"name": i, "id": i} for i in df_results_tackle.columns],
        #             style_data_conditional=discrete_background_color_bins(df_results_tackle, columns=['tackle_prob', 'tackle_prob_new', 'tackle_prob_delta']),
        #             id="tbl-results-tackle",
        #         ),
        #     ],
        #     style={"width": "60%", "float": "left", "display": "inline-block"},
        # ),
        html.Br(),
        html.Div(
            [
                dcc.Interval(id="animate", interval=200, disabled=True),
            ]
        )
    ]
)


# Change the play based on the dropdown selection
@callback(
    Output("slider-frame", "max"),
    Output("slider-frame", "marks"),
    Output("slider-frame", "value"),
    Output("plot-field", "figure"),
    Output("score", "children"),
    Output("down-distance", "children"),
    Output("time", "children"),
    Output("quarter", "children"),
    Output("possession", "src"),
    Output("animate", "disabled", allow_duplicate=True),
    Input("play-selection", "value"),
    prevent_initial_call=True,
)
def change_play(playId):
    df_play = df_plays[(df_plays.gameId == gameId) & (df_plays.playId == playId)]

    df_tracking_play = weekly_data[
        (weekly_data.gameId == gameId) & (weekly_data.playId == playId)
    ]

    # df_tracking_play = df_model_results[(df_model_results.playId == playId)]
    frames = df_tracking_play["frameId"].unique()

    max = len(frames)
    marks = dict(
        zip(
            [x for x in range(2, len(frames) + 1)],
            [{"label": "{}".format(int(x))} for x in frames[1:]],
        )
    )
    value = 2

    if df_tracking_play.playDirection.values[0] == "right":
        line_of_scrimmage = df_play.absoluteYardlineNumber.values[0]
        first_down_line = (
            df_play.absoluteYardlineNumber.values[0] + df_play.yardsToGo.values[0]
        )
    else:
        line_of_scrimmage = 120 - df_play.absoluteYardlineNumber.values[0]
        first_down_line = (
            120 - df_play.absoluteYardlineNumber.values[0] + df_play.yardsToGo.values[0]
        )

    fig_interactive = go.Figure(
        data=plot_field(
            line_of_scrimmage=line_of_scrimmage, first_down_line=first_down_line
        ),
        layout=layout_interactive,
    )

    score = "{} - {}".format(df_play['preSnapHomeScore'].values[0], df_play['preSnapVisitorScore'].values[0])
    down_distance = "{} & {}".format(df_play['down'].values[0], df_play['yardsToGo'].values[0])
    time = "{}".format(df_play['gameClock'].values[0])
    quarter = '4th Quarter' if df_play['quarter'].values[0] == 4 else '1st Quarter' if df_play['quarter'].values[0] == 1 else '2nd Quarter' if df_play['quarter'].values[0] == 2 else '3rd Quarter'
    possession = df_teams[df_teams['team_abbr'] == df_play['possessionTeam'].values[0]]['team_logo_wikipedia'].values[0]

    return max, marks, value, fig_interactive, score, down_distance, time, quarter, possession, True


# Change the Dash Cytoscape Graph based on the slider selection and the play selection
@callback(
    Output("cytoscape-test", "elements", allow_duplicate=True),
    Output("btn-click", "n_clicks"),
    Output("tbl-results-bc", "data", allow_duplicate=True),
    Output("tbl-results-bc", "columns", allow_duplicate=True),
    Output("tbl-results-bc", "style_data_conditional", allow_duplicate=True),
    Output("tbl-results-tackle", "data", allow_duplicate=True),
    Output("tbl-results-tackle", "columns", allow_duplicate=True),
    Output("tbl-results-tackle", "style_data_conditional", allow_duplicate=True),
    Input("slider-frame", "value"),
    Input("play-selection", "value"),
    prevent_initial_call=True,
)
def change_frame(sliderValue, playId):
    df_tracking_play = df_weekly_data[
        (df_weekly_data.gameId == gameId) & (df_weekly_data.playId == playId)
    ]
    # df_tracking_play = df_model_results[(df_model_results.playId == playId)]
    frames = df_tracking_play["frameId"].unique()
    frameId = frames[sliderValue - 1]

    # df_frame = weekly_data[
    #     (weekly_data["gameId"] == gameId)
    #     & (weekly_data["playId"] == playId)
    #     & (weekly_data["frameId"] == frameId)
    # ]
    df_frame = df_tracking_play[df_tracking_play["frameId"] == frameId].copy()

    # df_model_results_frame = df_model_results[
    #     (df_model_results["gameId"] == gameId)
    #     & (df_model_results["playId"] == playId)
    #     & (df_model_results["frameId"] == frameId)
    # ].copy()
    # df_model_results_frame["tackle_prob_new"] = df_model_results_frame["tackle_prob"]
    # df_model_results_frame["tackle_prob_delta"] = (
    #     df_model_results_frame["tackle_prob_new"]
    #     - df_model_results_frame["tackle_prob"]
    # )
    # df_model_results_frame["xYAC_new"] = df_model_results_frame["xYAC"]
    # df_model_results_frame["xYAC_delta"] = (
    #     df_model_results_frame["xYAC_new"] - df_model_results_frame["xYAC"]
    # )

    # df_model_results_frame = df_model_results_frame[
    #     df_model_results_frame["club"] != "football"
    # ]

    # df_results_bc = df_model_results_frame.loc[
    #     df_model_results_frame["isBallCarrier"] == 1,
    #     ["displayName", "jerseyNumber", "club", "xYAC", "xYAC_new", "xYAC_delta"],
    # ].copy()
    # df_results_bc = df_results_bc.round(2)
    # df_results_bc.rename(
    #     columns = {
    #         "displayName": "Player",
    #         "jerseyNumber": "Jersey Number",
    #         "club": "Team",
    #         "xYAC": "Expected Yards",
    #         "xYAC_new": "New Expected Yards",
    #         "xYAC_delta": "Delta Expected Yards",
    #     },
    #     inplace=True,
    # )

    # df_results_tackle = df_model_results_frame.loc[
    #     ~df_model_results_frame["tackle_prob"].isna(),
    #     [
    #         "displayName",
    #         "jerseyNumber",
    #         "club",
    #         "is_tackle",
    #         "tackle_prob",
    #         "tackle_prob_new",
    #         "tackle_prob_delta",
    #     ],
    # ].copy()
    # df_results_tackle = df_results_tackle.round(2)
    # df_results_tackle.rename(
    #     columns = {
    #         "displayName": "Player",
    #         "jerseyNumber": "Jersey Number",
    #         "club": "Team",
    #         "is_tackle": "Tackle",
    #         "tackle_prob": "Tackle Probability",
    #         "tackle_prob_new": "New Tackle Probability",
    #         "tackle_prob_delta": "Delta Tackle Probability",
    #     },
    #     inplace=True,
    # )
    # df_results_tackle.sort_values(by=["New Tackle Probability"], ascending=False, inplace=True)

    # style_bc = discrete_background_color_bins(df_results_bc, columns=['Expected Yards', 'New Expected Yards', 'Delta Expected Yards'])
    # style_tackle = discrete_background_color_bins(df_results_tackle, columns=['Tackle Probability', 'New Tackle Probability', 'Delta Tackle Probability'])

    # data_bc = df_results_bc.to_dict("records")
    # columns_bc = [{"name": i, "id": i} for i in df_results_bc.columns]
    # data_tackle = df_results_tackle.to_dict("records")
    # columns_tackle = [{"name": i, "id": i} for i in df_results_tackle.columns]

    # ball_carrier_id = df_model_results_frame["ballCarrierId"].values[0]
    # tackle_id = df_model_results_frame.loc[df_model_results_frame['is_tackle'] == 1, 'nflId'].values
    # data = frame_data(df_frame=df_frame, ball_carrier_id=ball_carrier_id, tackle_id=tackle_id, n_clicks=sliderValue)
    data = frame_data(df_frame=df_frame, ball_carrier_id=None, tackle_id=None, n_clicks=sliderValue)

    n_clicks = None

    # return data, n_clicks, data_bc, columns_bc, style_bc, data_tackle, columns_tackle, style_tackle
    return data, n_clicks, no_update, no_update, no_update, no_update, no_update, no_update



# # Iterate through the frames of the play
# @callback(
#     Output("cytoscape-test", "elements", allow_duplicate=True),
#     # Output("slider-frame", "value", allow_duplicate=True),
#     Input("btn-play", "n_clicks"),
#     # Input("slider-frame", "value"),
#     Input("play-selection", "value"),
#     prevent_initial_call=True,
# )
# def iterate_frames(n_clicks, playId):
#     if n_clicks:
#         gameId = 2022092900

#         df_tracking_play = df_model_results[(df_model_results.playId == playId)]
#         frames = df_tracking_play["frameId"].unique()
#         #frameId = frames[sliderValue - 1]
#         for frame in frames[5:]:
#             df_frame = df_tracking_play[df_tracking_play["frameId"] == frame]
#             ball_carrier_id = df_frame["ballCarrierId"].values[0]
#             tackle_id = df_frame.loc[df_frame['is_tackle'] == 1, 'nflId'].values
#             data = frame_data(df_frame=df_frame, ball_carrier_id=ball_carrier_id, tackle_id=tackle_id, n_clicks=frame)

#             return data

#     n_clicks = None

@app.callback(
    Output("slider-frame", "value", allow_duplicate=True),
    Input('animate', 'n_intervals'),
    State("play-selection", "value"),
    State("slider-frame", 'value'),
    prevent_initial_call=True,
)
def update_output(n, playId, sliderValue):
    df_tracking_play = df_weekly_data[
        (df_weekly_data.gameId == gameId) & (df_weekly_data.playId == playId)
    ]
    # df_tracking_play = df_model_results[(df_model_results.playId == playId)]
    frames = df_tracking_play["frameId"].unique()
    if sliderValue < len(frames):
        sliderValue += 1
    else:
        sliderValue = 2
    return sliderValue


@app.callback(
    Output("animate", "disabled", allow_duplicate=True),
    Input("btn-play", "n_clicks"),
    State("animate", "disabled"),
    prevent_initial_call=True,
)
def toggle(n, playing):
    if n:
        return not playing
    return playing

# Callback to the the Coordinates from the Dash Cytoscape Graph
@callback(
    # Output("output-coords", "children"),
    Output("tbl-results-bc", "data"),
    Output("tbl-results-bc", "columns"),
    Output("tbl-results-tackle", "data"),
    Output("tbl-results-tackle", "columns"),
    Input("btn-click", "n_clicks"),
    Input("slider-frame", "value"),
    Input("play-selection", "value"),
    State("cytoscape-test", "elements"),
    prevent_initial_call=True,
)
def get_coordinates(btn_click, sliderValue, playId, elements):
    if btn_click:
        gameId = 2022092900
        df_tracking_play = df_weekly_data[
            (df_weekly_data.gameId == gameId) & (df_weekly_data.playId == playId)
        ]
        # df_tracking_play = df_model_results[(df_model_results.playId == playId)]
        frames = df_tracking_play["frameId"].unique()
        frameId = frames[sliderValue - 1]

        df_frame = df_tracking_play[df_tracking_play["frameId"] == frameId]

        df_play = df_plays[(df_plays.gameId == gameId) & (df_plays.playId == playId)]

        df_test = create_data(elements=elements, df_frame=df_frame, scale=scale)

        # df_testing_new_frame = process_data_xyac(
        #     df_new_frame=df_test,
        #     df_play=df_play,
        #     df_players=players,
        # )
        # features_list = model_xgb_2.feature_names
        # results = model_xgb_2.predict(xgb.DMatrix(df_testing_new_frame[features_list]))

        # gameId, playId, frameId = df_test.iloc[0][["gameId", "playId", "frameId"]]
        # df_pursuit = df_tracking_play[df_tracking_play["frameId"] == (frameId - 1)]
        # df_pursuit = df_pursuit[['gameId', 'playId', 'nflId', 'displayName', 'frameId', 'time','jerseyNumber', 'club', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o','dir', 'event', 'second']].copy()
        # df_pursuit = pd.DataFrame(pd.concat([df_pursuit, df_test], axis=0))
        # df_test_bcddp = process_data_pursuit(
        #     df_frame=df_pursuit,
        #     df_play=df_play,
        #     df_players=players,
        # )

        # non_feats = ["gameId", "playId", "frameId", "nflId", "target"]
        # feats = [
        #     "x",
        #     "y",
        #     "sx",
        #     "sy",
        #     "ax",
        #     "ay",
        #     "x_bc",
        #     "y_bc",
        #     "sx_bc",
        #     "sy_bc",
        #     "ax_bc",
        #     "ay_bc",
        #     "p_bc_x",
        #     "p_bc_y",
        #     "p_bc_d",
        #     "p_bc_sx",
        #     "p_bc_sy",
        #     "p_bc_s_rel",
        #     "p_bc_s_rel_tt",
        #     "p_bc_ax",
        #     "p_bc_ay",
        #     "p_bc_a_rel",
        #     "p_bc_a_rel_tt",
        #     "p_bc_d_D_1",
        #     "p_bc_d_D_2",
        #     "p_bc_d_D_3",
        #     "p_bc_s_rel_D_1",
        #     "p_bc_s_rel_D_2",
        #     "p_bc_s_rel_D_3",
        #     "p_bc_s_rel_tt_D_1",
        #     "p_bc_s_rel_tt_D_2",
        #     "p_bc_s_rel_tt_D_3",
        #     "def_d_bc_ratio_sum",
        #     "def_o_bc_ratio_sum",
        # ]

        # df_bcddp_results = df_test_bcddp[non_feats].copy()
        # df_bcddp_results["pred_pursuit"] = model_xgb_pursuit.predict(
        #     xgb.DMatrix(df_test_bcddp[feats])
        # )
        # df_bcddp_results = df_bcddp_results.merge(
        #     df_test[["nflId", "displayName", "jerseyNumber", "club"]],
        #     on=["nflId"],
        #     how="left",
        # )

        # df_model_results_frame = df_model_results[
        #     (df_model_results["gameId"] == gameId)
        #     & (df_model_results["playId"] == playId)
        #     & (df_model_results["frameId"] == frameId)
        # ].copy()

        # df_model_results_frame = df_model_results_frame.merge(
        #     df_bcddp_results[["nflId", "pred_pursuit", "target"]],
        #     on=["nflId"],
        #     how="left",
        # )
        # df_model_results_frame.rename(
        #     columns={"pred_pursuit": "tackle_prob_new"}, inplace=True
        # )

        # df_model_results_frame["tackle_prob_delta"] = (
        #     df_model_results_frame["tackle_prob_new"]
        #     - df_model_results_frame["tackle_prob"]
        # )
        # df_model_results_frame["xYAC_new"] = round(results[0], 2)
        # df_model_results_frame["xYAC_delta"] = (
        #     df_model_results_frame["xYAC_new"] - df_model_results_frame["xYAC"]
        # )

        # df_model_results_frame = df_model_results_frame[
        #     df_model_results_frame["club"] != "football"
        # ]

        # df_results_bc = df_model_results_frame.loc[
        #     df_model_results_frame["isBallCarrier"] == 1,
        #     ["displayName", "jerseyNumber", "club", "xYAC", "xYAC_new", "xYAC_delta"],
        # ].copy()
        # df_results_bc["xYAC_new"] = df_results_bc["xYAC_new"].astype("float64")
        # df_results_bc = df_results_bc.round(2)
        # df_results_bc.rename(
        #     columns = {
        #         "displayName": "Player",
        #         "jerseyNumber": "Jersey Number",
        #         "club": "Team",
        #         "xYAC": "Expected Yards",
        #         "xYAC_new": "New Expected Yards",
        #         "xYAC_delta": "Delta Expected Yards",
        #     },
        #     inplace=True,
        # )


        # df_results_tackle = df_model_results_frame.loc[
        #     ~df_model_results_frame["tackle_prob"].isna(),
        #     [
        #         "displayName",
        #         "jerseyNumber",
        #         "club",
        #         "is_tackle",
        #         "tackle_prob",
        #         "tackle_prob_new",
        #         "tackle_prob_delta",
        #     ],
        # ].copy()
        # df_results_tackle["tackle_prob_new"] = df_results_tackle[
        #     "tackle_prob_new"
        # ].astype("float64")
        # df_results_tackle = df_results_tackle.round(2)
        # df_results_tackle.rename(
        #     columns = {
        #         "displayName": "Player",
        #         "jerseyNumber": "Jersey Number",
        #         "club": "Team",
        #         "is_tackle": "Tackle",
        #         "tackle_prob": "Tackle Probability",
        #         "tackle_prob_new": "New Tackle Probability",
        #         "tackle_prob_delta": "Delta Tackle Probability",
        #     },
        #     inplace=True,
        # )
        # df_results_tackle.sort_values(by=["New Tackle Probability"], ascending=False, inplace=True)

        # data_bc = df_results_bc.to_dict("records")
        # columns_bc = [{"name": i, "id": i} for i in df_results_bc.columns]
        # data_tackle = df_results_tackle.to_dict("records")
        # columns_tackle = [{"name": i, "id": i} for i in df_results_tackle.columns]

        # return data_bc, columns_bc, data_tackle, columns_tackle
        return no_update, no_update, no_update, no_update
    else:
        return no_update, no_update, no_update, no_update


if __name__ == "__main__":
    app.run(debug=True)
