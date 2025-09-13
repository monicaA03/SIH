import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 5 seconds ---
st_autorefresh(interval=5000, key="data_refresh")

# --- Define Stations and Tracks (Mumbai local example) ---
stations = {
    "CST": [18.9402, 72.8356],
    "Dadar": [19.0184, 72.8425],
    "Kurla": [19.0707, 72.8826],
    "Ghatkopar": [19.0790, 72.9080],
}
tracks = [
    ("CST", "Dadar"),
    ("Dadar", "Kurla"),
    ("Kurla", "Ghatkopar"),]

# --- Initialize train data in session state ---
if "df" not in st.session_state:
    trains_data = [
        {"train_id": "T101", "type": "Passenger", "priority": 3, "location": "CST", "eta": "12:05", "delay": 67, "section_start": "CST", "section_end": "Dadar", "progress": 0.1},
        {"train_id": "T102", "type": "Freight",   "priority": 1, "location": "Dadar", "eta": "12:07", "delay": 67, "section_start": "Dadar", "section_end": "Kurla", "progress": 0.4},
        {"train_id": "T103", "type": "Express",   "priority": 4, "location": "Kurla", "eta": "12:06", "delay": 87, "section_start": "Kurla", "section_end": "Ghatkopar", "progress": 0.2},
        {"train_id": "T104", "type": "Passenger", "priority": 2, "location": "Dadar", "eta": "12:10", "delay": 99, "section_start": "CST", "section_end": "Dadar", "progress": 0.9},
        {"train_id": "T105", "type": "Express",   "priority": 5, "location": "Kurla", "eta": "12:08", "delay": 88, "section_start": "Dadar", "section_end": "Kurla", "progress": 0.8},
    ]
    st.session_state.df = pd.DataFrame(trains_data)
    st.session_state.log = []

df = st.session_state.df.copy()df = df.copy()
    df["progress"] += np.random.uniform(0.02, 0.05, size=len(df))
    df["progress"] = df["progress"].apply(lambda x: x if x <= 1 else x - 1)

    # Move to next section if progress loops
    for i, row in df.iterrows():
        if row["progress"] <= 0:
            continue
        if row["progress"] == 0 or row["progress"] > 0.99:
            # Find index of current segment
            try:
                idx = tracks.index((row["section_start"], row["section_end"]))
                next_idx = (idx + 1) % len(tracks)
                df.at[i, "section_start"], df.at[i, "section_end"] = tracks[next_idx]
                df.at[i, "progress"] = 0.0
            except ValueError:
                pass
    return df

df = simulate_train_movement(df)

# --- Update location column for conflict detection ---
df["location"] = df["section_start"]

# --- Compute lat/lon by interpolating between section start and end stations ---
def interpolate_position(row):
    start_coord = stations[row["section_start"]]
    end_coord = stations[row["section_end"]]
    lat = start_coord[0] + (end_coord[0] - start_coord[0]) * row["progress"]
    lon = start_coord[1] + (end_coord[1] - start_coord[1]) * row["progress"]
    return pd.Series([lat, lon])

df[["lat", "lon"]] = df.apply(interpolate_position, axis=1)

st.session_state.df = df.copy()  # Update session state

# --- Conflict Detection ---
def detect_conflicts(df_in):
    conflicts = []
    for section, group in df_in.groupby("location"):
        if len(group) > 1:
            conflicts.append((section, group["train_id"].tolist()))
    return conflicts

conflicts = detect_conflicts(df)

# --- Recommendation Engine ---
def recommend_action(conflicts_in, df_in):
    recs = []
    for section, trains in conflicts_in:
        subset = df_in[df_in["train_id"].isin(trains)]
 winner_row = subset.sort_values(by=["priority", "eta"], ascending=[False, True])
        winner_id = winner_row.iloc[0]["train_id"]
        losers = [t for t in trains if t != winner_id]
        for t in losers:
            recs.append({
                "train_id": t,
                "priority": int(df_in.loc[df_in["train_id"] == t, "priority"].values[0]),
                "winner": winner_id,
                "section": section,
                "action": f"Hold {t} at {section} to allow {winner_id} first"
            })
    return recs

recommendations = recommend_action(conflicts, df)

# --- Apply Best Recommendation ---
def choose_best_and_update(recs):
    if not recs:
        return None, st.session_state.df
    best = sorted(recs, key=lambda x: x["priority"])[0]
    best["reason"] = f"{best['train_id']} has lower priority than {best['winner']}."
    held = best["train_id"]
    # Add 5 min delay and update ETA for held train
st.session_state.df.loc[st.session_state.df["train_id"] == held, "delay"] += 5
    eta_str = st.session_state.df.loc[st.session_state.df["train_id"] == held, "eta"].values[0]
    try:
        eta_time = datetime.datetime.strptime(eta_str, "%H:%M") + datetime.timedelta(minutes=5)
        st.session_state.df.loc[st.session_state.df["train_id"] == held, "eta"] = eta_time.strftime("%H:%M")
    except Exception:
        pass
    return best, st.session_state.df

best_rec, _ = choose_best_and_update(recommendations)

# --- KPIs ---
st.title("🚆 Railway Decision-Support System with Live Train Track Map")
col1, col2, col3 = st.columns(3)
col1.metric("🚆 Number of Trains", len(df))
col2.metric("⚠ Number of Conflicts", len(conflicts))
col3.metric("⏱ Average Delay (min)", round(df["delay"].mean(), 1))

# --- Delay Status for UI ---
def delay_status(delay):
    if delay < 5:
        return "✅ On Time"
    elif delay < 15:
  return "⚠ Minor Delay"
    else:
        return "❌ Major Delay"

df["status"] = df["delay"].apply(delay_status)

# --- Realistic Railway Track Map using Pydeck ---
track_layer = pdk.Layer(
    "LineLayer",
    pd.DataFrame([{
        "path": [stations[start], stations[end]],
        "color": [0, 128, 255],
        "width": 5,
    } for start, end in tracks]),
    get_path="path",
    get_color="color",
    get_width="width"
)

train_layer = pdk.Layer(
    "ScatterplotLayer",
    df,
    get_position=["lon", "lat"],
    get_color=[255, 0, 0],
    get_radius=100,
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=19.03,
    longitude=72.87,
    zoom=11,
    pitch=0,
)
deck = pdk.Deck(
    layers=[track_layer, train_layer],
    initial_view_state=view_state,
    tooltip={"text": "Train ID: {train_id}\nDelay: {delay} min\nETA: {eta}"}
)

st.subheader("🗺 Real-Time Railway Track & Train Map")
st.pydeck_chart(deck)

# --- Live Train Table ---
st.subheader("🚉 Live Train Status")
st.dataframe(df[["train_id", "type", "priority", "location", "eta", "delay", "status"]], use_container_width=True)

# --- Delay Chart (Sidebar) ---
def delay_color(delay):
    if delay < 5:
        return "green"
    elif delay < 15:
        return "orange"
    else:
        return "red"

df["color"] = df["delay"].apply(delay_color)
st.sidebar.subheader("📊 Delay by Train")
delay_chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("train_id:N", title="Train"),
        y=alt.Y("delay:Q", title="Delay (min)"),
   color=alt.Color("color:N", scale=None, legend=None),
        tooltip=["train_id", "eta", "delay", "status"]
    )
    .properties(width=250, height=180)
)
st.sidebar.altair_chart(delay_chart, use_container_width=True)

# --- Conflict Alerts and Recommendations ---
st.markdown("## 🚨 Conflict Alerts")
if conflicts:
    for sec, trains in conflicts:
        st.error(f"⚠ Section *{sec}* — conflict between: *{', '.join(trains)}*")
else:
    st.success("✅ No conflicts detected")

st.markdown("## 🧠 System Recommendation")
if best_rec:
    st.info(f"💡 *Auto-selected:* {best_rec['action']}")
    st.success(f"📌 Reason: {best_rec['reason']}")
    # Audit log entry
    if not st.session_state.log or st.session_state.log[-1].get("train_id") != best_rec["train_id"]:
        st.session_state.log.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "train_id": best_rec["train_id"],
            "decision": "Auto-Selected",
  "reason": best_rec["reason"],
            "new_eta": st.session_state.df.loc[st.session_state.df["train_id"] == best_rec["train_id"], "eta"].values[0]
        })
    # Manual override button
    if st.button(f"❌ Override {best_rec['train_id']}", key=f"ovr_{best_rec['train_id']}"):
        st.session_state.log.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "train_id": best_rec["train_id"],
            "decision": "Manual Override",
            "reason": "Controller input"
        })
else:
    st.info("✅ No system recommendation at the moment")

# --- Audit Log ---
st.subheader("📜 Audit Log")
if st.session_state.log:
    st.dataframe(pd.DataFrame(st.session_state.log))
else:
    st.text("No actions yet."
