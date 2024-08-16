from dash import Dash, html, callback, Input, Output, State, dcc, dash_table, no_update
import dash_cytoscape as cyto
import plotly.graph_objects as go
import numpy as np

colors = {"background": "#111111", "text": "#7FDBFF"}
app = Dash(__name__)

# define colors
color_football = ["#CBB67C", "#663831"]

# define scale for the field
scale = 12


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

line_of_scrimmage = 60
first_down_line = 100

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
    # {
    #     "selector": '[classes = "{}"]'.format(teams[0]),
    #     "style": {
    #         "width": 22.5,
    #         "height": 22.5,
    #         "background-color": df_teams.loc[
    #             df_teams["team_abbr"] == teams[0], "team_color"
    #         ].values[0],
    #         "border-color": df_teams.loc[
    #             df_teams["team_abbr"] == teams[0], "team_color2"
    #         ].values[0],
    #         "border-width": 2.5,
    #         "font-size": 12.5,
    #         "text-valign": "center",
    #         "text-halign": "center",
    #         "color": "white",
    #     },
    # },
    # {
    #     "selector": '[classes = "{}"]'.format(teams[1]),
    #     "style": {
    #         "width": 22.5,
    #         "height": 22.5,
    #         "background-color": df_teams.loc[
    #             df_teams["team_abbr"] == teams[1], "team_color"
    #         ].values[0],
    #         "border-color": df_teams.loc[
    #             df_teams["team_abbr"] == teams[1], "team_color2"
    #         ].values[0],
    #         "border-width": 2.5,
    #         "font-size": 12.5,
    #         "text-valign": "center",
    #         "text-halign": "center",
    #         "color": "white",
    #     },
    # },
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
        # html.Div(
        #     [
        #         dcc.Dropdown(
        #             options=df_game_plays[["playId", "playDescription"]]
        #             .sort_values("playId")
        #             .apply(
        #                 lambda x: {
        #                     "label": "{} - {}".format(
        #                         x["playId"], x["playDescription"]
        #                     ),
        #                     "value": x["playId"],
        #                 },
        #                 axis=1,
        #             )
        #             .values.tolist(),
        #             value=df_game_plays["playId"].sort_values().unique()[0],
        #             clearable=False,
        #             id="play-selection",
        #         ),
        #         html.Br(),
        #         html.Div(
        #             [
        #                 html.Button("Start/Stop Play", id="btn-play", style={"margin-left": "10px"}),
        #             ]
        #         ),
        #         html.Br(),
        #         dcc.Slider(
        #             min=2,
        #             max=len(initial_frames),
        #             step=1,
        #             value=2,
        #             marks=dict(
        #                 zip(
        #                     [x for x in range(2, len(initial_frames) + 1)],
        #                     [{"label": "{}".format(int(x))} for x in initial_frames[1:]],
        #                 )
        #             ),
        #             id="slider-frame",
        #         ),
        #     ],
        #     style={"width": "75%", "display": "inline-block"},
        # ),
        html.Br(),
        # html.Img(
        #     id='home-logo',
        #     src=df_teams[df_teams['team_abbr'] == 'CIN']['team_logo_wikipedia'].values[0],
        #     height=25,
        #     style={"margin-left": "10%"}
        # ),
        # html.H1(
        #     id="score",
        #     children="""{} - {}""".format(df_initial_play['preSnapHomeScore'].values[0], df_initial_play['preSnapVisitorScore'].values[0]),
        #     style={"float": "center", "display": "inline-block", "margin-left": "10px"},
        # ),
        # html.Img(
        #     id='away-logo',
        #     src=df_teams[df_teams['team_abbr'] == 'MIA']['team_logo_wikipedia'].values[0],
        #     height=25,
        #     style={"margin-left": "10px"}
        # ),
        # html.H1(
        #     id="down-distance",
        #     children="""{} & {}""".format(df_initial_play['down'].values[0], df_initial_play['yardsToGo'].values[0]),
        #     style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        # ),
        # html.H1(
        #     id="time",
        #     children="""{}""".format(df_initial_play['gameClock'].values[0]),
        #     style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        # ),
        # html.H1(
        #     id="quarter",
        #     # children="""{}th Quarter""".format(df_initial_play['quarter'].values[0]),
        #     children='4th Quarter' if df_initial_play['quarter'].values[0] == 4 else '1st Quarter' if df_initial_play['quarter'].values[0] == 1 else '2nd Quarter' if df_initial_play['quarter'].values[0] == 2 else '3rd Quarter',
        #     style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        # ),
        html.H1(
            id="possession-name",
            children="""Possession Team: """,
            style={"float": "center", "display": "inline-block", "margin-left": "30px"},
        ),
        # html.Img(
        #     id='possession',
        #     src=df_teams[df_teams['team_abbr'] == df_initial_play['possessionTeam'].values[0]]['team_logo_wikipedia'].values[0],
        #     height=25,
        #     style={"margin-left": "10px"}
        # ),
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
                # html.Div(
                #     [
                #         cyto.Cytoscape(
                #             id="cytoscape-test",
                #             layout={
                #                 "name": "preset",
                #                 "fit": True,
                #             },
                #             style={
                #                 "position": "absolute",
                #                 "width": "{}px".format(120 * scale),
                #                 "height": "{}px".format(53.3 * scale),
                #             },
                #             elements=data,
                #             zoom=1,
                #             zoomingEnabled=False,
                #             panningEnabled=False,
                #             stylesheet=stylesheet,
                #         ),
                #     ]
                # ),
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

if __name__ == "__main__":
    app.run(debug=True)
