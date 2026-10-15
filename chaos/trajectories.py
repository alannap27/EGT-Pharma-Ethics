import scipy as sp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model import *

t_span = (0,400)
x0 = np.array([0.5,0.5,0.5])
sol = sp.integrate.solve_ivp(lambda t,x: fun(x,t), t_span, x0, rtol=1e-6)

sol_df = pd.DataFrame(sol.y.T, columns=["x","y","z"])
sol_df['t'] = sol.t

# Create cumulative trace
traces = []
for time in sol_df["t"]:
    traces.append(sol_df[sol_df["t"] <= time])

sol_df_cum = pd.concat(traces, keys=sol_df["t"].rename("frame")).reset_index(level="frame")

fig = px.line_3d(sol_df_cum, x='x', y='y', z='z', animation_frame='frame')
fig.update_layout(scene=dict(
    xaxis=dict(range=[0,1], autorange=False),
    yaxis=dict(range=[0,1], autorange=False),
    zaxis=dict(range=[0,1], autorange=False),
    aspectmode="cube",
    ))
fig.show()
