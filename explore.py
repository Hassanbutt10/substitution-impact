"""
Substitution Impact - Cycle 9, Day 6 project
A chart can show that teams change after a substitution, but the match clock
moves at the same time: fatigue, the score and the opponent all change too.
This project asks what changes after a substitution BEYOND what happens to
other teams at the same minute with no substitution.

Method:
  1. Periods 1 and 2 only. Match time is continuous: second-half time starts
     where the first half ended. Substitutions made within 2 minutes of each
     other count as one wave, anchored at the first one. Injury substitutions
     are left out of the waves when the data labels them.
  2. For every wave: change = the 10 minutes after minus the 10 minutes
     before, for five measures: shots for, shots against, pressures (all per
     10 minutes), field tilt (the team's share of final-third passes) and
     share of passes.
  3. A wave is "clean" only if, in the 20-minute window around it, there is no
     goal, no red card and no other substitution by the same team.
  4. Baseline: the same change measured for every other team-match at the
     same minute (within 5 minutes) that also had no goal, no red card and no
     own substitution in the window. This is what fatigue and the clock do
     without a substitution.
  5. Beyond baseline = the wave's change minus that baseline. Results are the
     mean over clean waves with a 5th to 95th percentile range from 5,000
     bootstrap resamples of the waves.
  6. A second chart splits the waves by the score at the moment of the change
     (leading, level, trailing).

Limitations: only 32 matches, so the number of clean waves is small and the
ranges are wide. The baseline removes the effect of the clock but not what a
manager saw before deciding to change. Who came on and who went off is not
used, only the timing. Pressures depend on how long the opponent has the ball.
Field tilt is a share of final-third passes, not of time. Level includes early
0-0 minutes, when teams are cautious.

Outputs: substitution_impact.png, substitution_by_state.png,
substitution_replay.mp4 (match replay plus the two charts as end cards),
substitution_waves.csv

Reuses: copa_america_2024_matches.csv, copa_america_2024_events.csv
"""

import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Circle
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

FOCUS_TEAM = "Argentina"
W = 10                 # minutes before and after
WAVE_GAP = 120         # seconds between substitutions in one wave
PLACEBO_RANGE = 5      # minutes either side when matching the baseline
MIN_PLACEBO = 10       # least baseline windows needed
N_BOOT = 5000
SEED = 21
DOT_WINDOW = 180
LINE_WINDOW = 300
N_FRAMES = 300
HOLD_FRAMES = 30
END_CARDS = ["substitution_impact.png", "substitution_by_state.png"]
END_CARD_SECONDS = 4

BG = "#0d1b2a"
GRID = "#3a4a5a"
BLUE = "#4a90d9"
ORANGE = "#f4a300"
MUTED = "#a8b8c8"
GREEN = "#6fcf97"
RED = "#e76f51"

METRICS = [("shots_for", "Shots for (per 10 min)"),
           ("shots_against", "Shots against (per 10 min)"),
           ("tilt", "Field tilt (%)"),
           ("press", "Pressures (per 10 min)"),
           ("pass_share", "Share of passes (%)")]
MKEYS = [m for m, _ in METRICS]

matches = pd.read_csv("copa_america_2024_matches.csv")
events = pd.read_csv("copa_america_2024_events.csv", low_memory=False)
for col in ["shot_type", "shot_outcome", "location", "minute", "second", "index",
            "substitution_outcome", "substitution_replacement", "foul_committed_card",
            "bad_behaviour_card", "player"]:
    if col not in events.columns:
        events[col] = np.nan

xy = events["location"].astype(str).str.extract(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")
events["lx"] = pd.to_numeric(xy[0], errors="coerce")
events["ly"] = pd.to_numeric(xy[1], errors="coerce")

# ---------- match clock, goals, cards ----------
ev = events[events["period"].isin([1, 2])].copy()
ev["minute"] = ev["minute"].fillna(0)
ev["second"] = ev["second"].fillna(0)
ev["t"] = ev["minute"] * 60 + ev["second"]
p1_len = ev[ev["period"] == 1].groupby("match_id")["t"].max()
ev["t"] = np.where(ev["period"] == 2,
                   ev["match_id"].map(p1_len).fillna(2700) + (ev["t"] - 2700), ev["t"])
sort_cols = ["match_id", "index"] if ev["index"].notna().any() else ["match_id", "t"]
ev = ev.sort_values(sort_cols).reset_index(drop=True)

home_of = matches.set_index("match_id")["home_team"]
away_of = matches.set_index("match_id")["away_team"]
ev["home"] = ev["match_id"].map(home_of)
ev["away"] = ev["match_id"].map(away_of)
ev = ev[ev["home"].notna()].reset_index(drop=True)
ev["k"] = (ev["t"] // 60).astype(int)

is_goal = ((ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")) | (ev["type"] == "Own Goal For")
ev["g_h"] = (is_goal & (ev["team"] == ev["home"])).astype(int)
ev["g_a"] = (is_goal & (ev["team"] == ev["away"])).astype(int)
cum_h = ev.groupby("match_id")["g_h"].cumsum() - ev["g_h"]
cum_a = ev.groupby("match_id")["g_a"].cumsum() - ev["g_a"]
ev["gd_home"] = cum_h - cum_a
is_red = (ev["foul_committed_card"].astype(str).str.contains("Red|Second Yellow", na=False)
          | ev["bad_behaviour_card"].astype(str).str.contains("Red|Second Yellow", na=False))

match_info = {}
for mid, g in ev.groupby("match_id"):
    t_end = float(g["t"].max())
    match_info[mid] = dict(
        home=g["home"].iloc[0], away=g["away"].iloc[0], t_end=t_end, T_bins=int(t_end // 60),
        goals=g.loc[(g["g_h"] + g["g_a"]) > 0, "t"].tolist(),
        reds=g.loc[is_red.loc[g.index], "t"].tolist())
opp_of = {}
for mid, info in match_info.items():
    opp_of[(mid, info["home"])] = info["away"]
    opp_of[(mid, info["away"])] = info["home"]

# ---------- minute bins per team ----------
SERIES = ["shots", "press", "pass", "tilt"]
arr = {}
for mid, info in match_info.items():
    L = info["T_bins"] + 2
    for team in (info["home"], info["away"]):
        arr[(mid, team)] = {s: np.zeros(L) for s in SERIES}
masks = {"shots": (ev["type"] == "Shot") & (ev["shot_type"].fillna("") != "Penalty"),
         "press": ev["type"] == "Pressure",
         "pass": ev["type"] == "Pass",
         "tilt": (ev["type"] == "Pass") & (ev["lx"] >= 80)}
for s, m in masks.items():
    g = ev[m].groupby(["match_id", "team", "k"]).size()
    for (mid, team, k), v in g.items():
        a = arr.get((mid, team))
        if a is not None and k < len(a[s]):
            a[s][k] += v


def seg(mid, team, a, b):
    me = arr[(mid, team)]
    op = arr[(mid, opp_of[(mid, team)])]
    tf, ta = me["tilt"][a:b].sum(), op["tilt"][a:b].sum()
    pf, pa = me["pass"][a:b].sum(), op["pass"][a:b].sum()
    return dict(shots_for=me["shots"][a:b].sum(), shots_against=op["shots"][a:b].sum(),
                tilt=tf / (tf + ta) * 100 if tf + ta >= 5 else np.nan,
                press=me["press"][a:b].sum(),
                pass_share=pf / (pf + pa) * 100 if pf + pa >= 20 else np.nan)


def change(mid, team, k):
    pre, post = seg(mid, team, k - W, k), seg(mid, team, k, k + W)
    return {m: post[m] - pre[m] for m in MKEYS}, pre, post


# ---------- substitutions and waves ----------
subs = ev[ev["type"] == "Substitution"].copy()
tactical = ~subs["substitution_outcome"].astype(str).str.contains("Injury", na=False)
sub_all = {}
for (mid, team), g in subs.groupby(["match_id", "team"]):
    sub_all[(mid, team)] = g["t"].tolist()
sub_tac = subs[tactical]

waves = []
for (mid, team), g in sub_tac.groupby(["match_id", "team"]):
    info = match_info[mid]
    gg = g.sort_values("t")
    clusters, cur = [], []
    for r in gg.itertuples():
        if cur and r.t - cur[-1].t > WAVE_GAP:
            clusters.append(cur)
            cur = []
        cur.append(r)
    if cur:
        clusters.append(cur)
    for c in clusters:
        t0, t_last = c[0].t, c[-1].t
        k = int(t0 // 60)
        gd = c[0].gd_home if team == info["home"] else -c[0].gd_home
        row = dict(match_id=mid, team=team, opp=opp_of[(mid, team)], t=t0, minute=int(k),
                   n_subs=len(c), state=int(np.sign(gd)))
        valid = k - W >= 0 and k + W <= info["T_bins"]
        row["valid"] = valid
        if valid:
            lo, hi = (k - W) * 60, (k + W) * 60
            goal_in = any(lo <= x < hi for x in info["goals"])
            red_in = any(lo <= x < hi for x in info["reds"])
            other_sub = any(lo <= x < hi and not (t0 <= x <= t_last) for x in sub_all.get((mid, team), []))
            row["goal_in_window"] = goal_in
            row["clean"] = not (goal_in or red_in or other_sub)
            ch, pre, post = change(mid, team, k)
            for m in MKEYS:
                row[m] = ch[m]
                row["pre_" + m] = pre[m]
                row["post_" + m] = post[m]
        else:
            row["goal_in_window"] = False
            row["clean"] = False
            for m in MKEYS:
                row[m] = np.nan
                row["pre_" + m] = np.nan
                row["post_" + m] = np.nan
        waves.append(row)
wv = pd.DataFrame(waves)
if wv.empty:
    raise ValueError("No substitution events found in the events file")

# ---------- baseline: same minute, no substitution, no goal, no red card ----------
MAXB = max(i["T_bins"] for i in match_info.values()) + 2
plc_sum = {m: np.zeros(MAXB) for m in MKEYS}
plc_cnt = {m: np.zeros(MAXB) for m in MKEYS}
for (mid, team) in arr:
    info = match_info[mid]
    own_subs = sub_all.get((mid, team), [])
    blockers = own_subs + info["goals"] + info["reds"]
    for k in range(W, info["T_bins"] - W + 1):
        lo, hi = (k - W) * 60, (k + W) * 60
        if any(lo <= x < hi for x in blockers):
            continue
        ch, _, _ = change(mid, team, k)
        for m in MKEYS:
            if np.isfinite(ch[m]):
                plc_sum[m][k] += ch[m]
                plc_cnt[m][k] += 1


def baseline(m, k):
    a, b = max(0, k - PLACEBO_RANGE), min(MAXB, k + PLACEBO_RANGE + 1)
    c = plc_cnt[m][a:b].sum()
    return plc_sum[m][a:b].sum() / c if c >= MIN_PLACEBO else np.nan


for m in MKEYS:
    wv["base_" + m] = [baseline(m, int(k)) for k in wv["minute"]]
    wv["adj_" + m] = wv[m] - wv["base_" + m]
wv.to_csv("substitution_waves.csv", index=False)

cw = wv[wv["clean"]].copy()
state_name = {1: "leading", 0: "level", -1: "trailing"}
print(f"\nSubstitution waves found: {len(wv)} (tactical changes only when the data labels injuries)")
print(f"  with a full 10 minutes before and after: {int(wv['valid'].sum())}")
print(f"  clean (no goal, red card or other own substitution in the window): {len(cw)}")
print(f"  baseline windows per measure: {int(plc_cnt['shots_for'].sum())}")
for s in (1, 0, -1):
    print(f"  clean waves while {state_name[s]}: {int((cw['state'] == s).sum())}")

rng = np.random.default_rng(SEED)


def boot(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    if n < 5:
        return float(x.mean()), np.nan, np.nan, n
    bm = rng.choice(x, (N_BOOT, n)).mean(axis=1)
    lo, hi = np.percentile(bm, [5, 95])
    return float(x.mean()), float(lo), float(hi), n


res = {}
print("\nChange after a clean substitution wave (10 minutes after minus 10 before):")
print(f"  {'measure':<30}{'raw change':>12}{'beyond baseline':>34}")
for m, lab in METRICS:
    raw = boot(cw[m])
    adj = boot(cw["adj_" + m])
    res[m] = (raw, adj)
    print(f"  {lab:<30}{raw[0]:>+12.2f}   {adj[0]:>+8.2f}  ({adj[1]:+.2f} to {adj[2]:+.2f})  n={adj[3]}")

fm = wv[wv["team"] == FOCUS_TEAM]
print(f"\n{FOCUS_TEAM}: {len(fm)} substitution waves, {int(fm['clean'].sum())} clean")

# ---------- chart 1: raw vs beyond baseline ----------
fig1, axes = plt.subplots(2, 3, figsize=(11, 8))
fig1.patch.set_facecolor(BG)
flat = axes.ravel()
for ax, (m, lab) in zip(flat[:5], METRICS):
    ax.set_facecolor(BG)
    raw, adj = res[m]
    vals = [0.0]
    for xi, (r, col) in enumerate(((raw, BLUE), (adj, ORANGE))):
        mean, lo, hi, n = r
        if not np.isfinite(mean):
            continue
        lo_ = lo if np.isfinite(lo) else mean
        hi_ = hi if np.isfinite(hi) else mean
        ax.bar(xi, mean, width=0.6, color=col)
        ax.errorbar(xi, mean, yerr=[[mean - lo_], [hi_ - mean]], fmt="none", ecolor="white",
                    capsize=4, linewidth=1.3)
        vals += [lo_, hi_, mean]
    span = max(max(vals) - min(vals), 1e-6)
    ax.set_ylim(min(vals) - span * 0.25, max(vals) + span * 0.25)
    for xi, r in enumerate((raw, adj)):
        mean, lo, hi, n = r
        if not np.isfinite(mean):
            continue
        lo_ = lo if np.isfinite(lo) else mean
        hi_ = hi if np.isfinite(hi) else mean
        yy = hi_ + span * 0.04 if mean >= 0 else lo_ - span * 0.04
        ax.text(xi, yy, f"{mean:+.2f}", ha="center", va="bottom" if mean >= 0 else "top",
                color="white", fontsize=10.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw\nchange", "Beyond the\nsame-minute baseline"], color="white", fontsize=9)
    ax.axhline(0, color="white", linewidth=0.9)
    ax.set_title(lab, color="white", fontsize=10.5, pad=6)
    ax.tick_params(axis="y", colors="white", labelsize=9)
    ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)
note = flat[5]
note.axis("off")
note.text(0.0, 0.95, f"{len(cw)} clean substitution waves", color="white", fontsize=12, fontweight="bold", va="top")
note.text(0.0, 0.80, "Clean: no goal, red card or other\nsubstitution by the same team in the\n20-minute window around the change.",
          color=MUTED, fontsize=9.5, va="top")
note.text(0.0, 0.48, "Baseline: the same change for teams at\nthe same minute with no substitution.\nIt removes what the clock does.",
          color=MUTED, fontsize=9.5, va="top")
fig1.suptitle("What Changes After a Substitution?", color="white", fontsize=17, fontweight="bold", y=0.98)
fig1.text(0.5, 0.925, "10 minutes after minus 10 minutes before. Copa America 2024, first 90 minutes",
          ha="center", color=MUTED, fontsize=10.5)
fig1.text(0.5, 0.012, "Bars: mean over waves. Whiskers: 5th to 95th percentile (5,000 resamples).",
          ha="center", color=MUTED, fontsize=9.5)
plt.tight_layout(rect=[0, 0.03, 1, 0.91])
fig1.savefig("substitution_impact.png", dpi=150, facecolor=BG)
plt.close(fig1)
print("\nSaved chart: substitution_impact.png")

# ---------- chart 2: by score state at the change ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 7.5))
fig2.patch.set_facecolor(BG)
S_ORDER = [(1, "Leading", GREEN), (0, "Level", MUTED), (-1, "Trailing", RED)]
for ax, (m, lab) in zip(axes2, [("tilt", "Field tilt (%), beyond baseline"),
                                 ("shots_for", "Shots for per 10 min, beyond baseline")]):
    ax.set_facecolor(BG)
    vals, labels, stats = [0.0], [], []
    for xi, (s, nm, col) in enumerate(S_ORDER):
        sub = cw[cw["state"] == s]
        mean, lo, hi, n = boot(sub["adj_" + m])
        labels.append(f"{nm}\n(n={n})")
        stats.append((mean, lo, hi, n))
        if not np.isfinite(mean):
            continue
        lo_ = lo if np.isfinite(lo) else mean
        hi_ = hi if np.isfinite(hi) else mean
        ax.bar(xi, mean, width=0.6, color=col)
        ax.errorbar(xi, mean, yerr=[[mean - lo_], [hi_ - mean]], fmt="none", ecolor="white",
                    capsize=4, linewidth=1.3)
        vals += [lo_, hi_, mean]
    span = max(max(vals) - min(vals), 1e-6)
    ax.set_ylim(min(vals) - span * 0.25, max(vals) + span * 0.25)
    for xi, (mean, lo, hi, n) in enumerate(stats):
        if not np.isfinite(mean):
            continue
        lo_ = lo if np.isfinite(lo) else mean
        hi_ = hi if np.isfinite(hi) else mean
        yy = hi_ + span * 0.04 if mean >= 0 else lo_ - span * 0.04
        ax.text(xi, yy, f"{mean:+.2f}", ha="center", va="bottom" if mean >= 0 else "top",
                color="white", fontsize=11)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, color="white", fontsize=10.5)
    ax.axhline(0, color="white", linewidth=0.9)
    ax.set_title(lab, color="white", fontsize=12, pad=8)
    ax.tick_params(axis="y", colors="white")
    ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)
fig2.suptitle("Does the Score Change What a Substitution Does?", color="white", fontsize=17,
              fontweight="bold", y=0.97)
fig2.text(0.5, 0.905, "Clean substitution waves split by the score at the moment of the change. Copa America 2024",
          ha="center", color=MUTED, fontsize=10.5)
fig2.text(0.5, 0.02, "Whiskers: 5th to 95th percentile. No whiskers when a group has fewer than 5 waves.",
          ha="center", color=MUTED, fontsize=9.5)
plt.tight_layout(rect=[0, 0.05, 1, 0.9])
fig2.savefig("substitution_by_state.png", dpi=150, facecolor=BG)
plt.close(fig2)
print("Saved chart: substitution_by_state.png")

# ---------- choose a match to replay ----------
best_mid, best_key = None, (-1, -1)
for mid, g in wv[wv["team"] == FOCUS_TEAM].groupby("match_id"):
    key = (int(g["clean"].sum()), len(g))
    if key > best_key:
        best_mid, best_key = mid, key
if best_mid is None:
    raise ValueError(f"No {FOCUS_TEAM} substitution waves found to replay")
info = match_info[best_mid]
F, O = FOCUS_TEAM, opp_of[(best_mid, FOCUS_TEAM)]
home, away = info["home"], info["away"]
t_end = info["t_end"]
mev = ev[ev["match_id"] == best_mid].reset_index(drop=True)
P1 = float(p1_len.get(best_mid, 2700.0))
print(f"\nReplaying {F} vs {O}: {best_key[1]} substitution waves, {best_key[0]} clean")

fw = wv[(wv["match_id"] == best_mid) & (wv["team"] == F)].sort_values("t").to_dict("records")
ow = wv[(wv["match_id"] == best_mid) & (wv["team"] == O)].sort_values("t").to_dict("records")
goal_rows = mev[(mev["g_h"] + mev["g_a"]) > 0][["t", "g_h", "g_a"]]
goals = [(float(r.t), home if r.g_h else away) for r in goal_rows.itertuples()]


def passes_of(team, flip):
    d = mev[(mev["team"] == team) & (mev["type"] == "Pass") & mev["lx"].notna() & mev["ly"].notna()]
    out = pd.DataFrame({"t": d["t"].to_numpy(), "x": d["lx"].to_numpy(), "y": d["ly"].to_numpy()})
    if flip:
        out["x"] = 120 - out["x"]
        out["y"] = 80 - out["y"]
    return out


pf_df, po_df = passes_of(F, False), passes_of(O, True)
T_grid = np.linspace(0, t_end, N_FRAMES)


def roll_avg(df, T, flip):
    w = df[(df["t"] > T - LINE_WINDOW) & (df["t"] <= T)]
    if len(w) < 5:
        return np.nan
    m = float(w["x"].mean())
    return 120 - m if flip else m


f_roll = np.array([roll_avg(pf_df, T, False) for T in T_grid])
o_roll = np.array([roll_avg(po_df, T, True) for T in T_grid])
all_roll = np.concatenate([f_roll[np.isfinite(f_roll)], o_roll[np.isfinite(o_roll)]])
y_lo, y_hi = (all_roll.min() - 4, all_roll.max() + 4) if len(all_roll) else (30, 80)


def disp_min(T):
    return int(T / 60) if T <= P1 else int(45 + (T - P1) / 60)


def score_at(T):
    fs = sum(1 for g in goals if g[0] <= T and g[1] == F)
    os_ = sum(1 for g in goals if g[0] <= T and g[1] == O)
    return fs, os_


def fmt(v, d=0):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


tick_t = [0, 900, 1800, P1, P1 + 900, P1 + 1800, P1 + 2700]
tick_l = ["0'", "15'", "30'", "45'", "60'", "75'", "90'"]


def draw_pitch(ax):
    ax.set_facecolor("#10253a")
    kw = dict(fill=False, edgecolor="#35506b", linewidth=1.6)
    ax.add_patch(Rectangle((0, 0), 120, 80, **kw))
    ax.plot([60, 60], [0, 80], color="#35506b", linewidth=1.6)
    ax.add_patch(Circle((60, 40), 10, **kw))
    ax.add_patch(Rectangle((0, 18), 18, 44, **kw))
    ax.add_patch(Rectangle((102, 18), 18, 44, **kw))
    ax.add_patch(Rectangle((0, 30), 6, 20, **kw))
    ax.add_patch(Rectangle((114, 30), 6, 20, **kw))
    ax.set_xlim(-2, 122)
    ax.set_ylim(82, -2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


fig = plt.figure(figsize=(10, 10), dpi=108)  # 1080 x 1080 pixels
fig.patch.set_facecolor(BG)
axp = fig.add_axes([0.12, 0.38, 0.76, 0.45])
axt = fig.add_axes([0.10, 0.07, 0.82, 0.15])
fig.text(0.5, 0.965, "What Changes After a Substitution?", ha="center",
         color="white", fontsize=23, fontweight="bold")
fig.text(0.5, 0.928, f"{F} vs {O}, Copa America 2024", ha="center", color=MUTED, fontsize=14)
left_t = fig.text(0.47, 0.882, "", ha="right", color=BLUE, fontsize=26, fontweight="bold")
fig.text(0.5, 0.882, "-", ha="center", color="white", fontsize=26, fontweight="bold")
right_t = fig.text(0.53, 0.882, "", ha="left", color=ORANGE, fontsize=26, fontweight="bold")
state_t = fig.text(0.5, 0.845, "", ha="center", color="white", fontsize=17, fontweight="bold")
note_t = fig.text(0.5, 0.358, f"Dots: passes in the last 3 minutes. {F} attacks right, {O} attacks left.",
                  ha="center", color=MUTED, fontsize=10.5)
sub1_t = fig.text(0.5, 0.322, "", ha="center", color="white", fontsize=16, fontweight="bold")
sub2_t = fig.text(0.5, 0.288, "", ha="center", color=GREEN, fontsize=15)
sub3_t = fig.text(0.5, 0.255, "", ha="center", color=GREEN, fontsize=15)
fig.text(0.5, 0.012, "StatsBomb open data, Copa America 2024.", ha="center", color=MUTED, fontsize=9)


def dots(ax, df, T, col):
    w = df[(df["t"] > T - DOT_WINDOW) & (df["t"] <= T)]
    if w.empty:
        return
    age = (T - w["t"].to_numpy()) / DOT_WINDOW
    cols = [to_rgba(col, float(0.95 - 0.8 * a)) for a in age]
    ax.scatter(w["x"], w["y"], s=46, c=cols, edgecolors="none", zorder=3)


def draw(i):
    k = min(i, N_FRAMES - 1)
    T = T_grid[k]
    axp.clear()
    axt.clear()
    draw_pitch(axp)
    dots(axp, pf_df, T, BLUE)
    dots(axp, po_df, T, ORANGE)
    if np.isfinite(f_roll[k]):
        axp.axvline(f_roll[k], color=BLUE, linewidth=3, linestyle="--", zorder=4)
    if np.isfinite(o_roll[k]):
        axp.axvline(120 - o_roll[k], color=ORANGE, linewidth=3, linestyle="--", zorder=4)
    fs, os_ = score_at(T)
    left_t.set_text(f"{F} {fs}")
    right_t.set_text(f"{os_} {O}")
    diff = fs - os_
    word, col = (("leading", GREEN) if diff > 0 else (("trailing", RED) if diff < 0 else ("level", MUTED)))
    state_t.set_text(f"{disp_min(T)}'   |   {F} {word}")
    state_t.set_color(col)

    # substitution information for the latest wave
    done = [w for w in fw if w["t"] <= T]
    if not done:
        sub1_t.set_text("No substitutions yet")
        sub1_t.set_color(MUTED)
        sub2_t.set_text("")
        sub3_t.set_text("")
    else:
        lw = done[-1]
        mnt = int(lw["t"] // 60) if lw["t"] <= P1 else int(45 + (lw["t"] - P1) / 60)
        plural = "s" if lw["n_subs"] > 1 else ""
        sub1_t.set_text(f"{F} substitution wave at {mnt}' ({lw['n_subs']} change{plural})")
        sub1_t.set_color("white")
        if not lw["valid"]:
            sub2_t.set_text("Not enough match left to measure 10 minutes before and after")
            sub2_t.set_color(MUTED)
            sub3_t.set_text("")
        elif T < lw["t"] + W * 60:
            left_min = int((lw["t"] + W * 60 - T) / 60) + 1
            sub2_t.set_text(f"Measuring the next 10 minutes ({left_min} left)")
            sub2_t.set_color(ORANGE)
            sub3_t.set_text(f"Field tilt before: {fmt(lw['pre_tilt'])}%   Shots for before: {fmt(lw['pre_shots_for'])}")
            sub3_t.set_color(MUTED)
        else:
            extra = ("   (goal in the window)" if lw["goal_in_window"]
                     else ("" if lw["clean"] else "   (other changes in the window)"))
            sub2_t.set_text(f"Field tilt {fmt(lw['pre_tilt'])}% to {fmt(lw['post_tilt'])}%    "
                            f"Shots for {fmt(lw['pre_shots_for'])} to {fmt(lw['post_shots_for'])}")
            sub2_t.set_color(GREEN)
            sub3_t.set_text(f"Pressures {fmt(lw['pre_press'])} to {fmt(lw['post_press'])}    "
                            f"Shots against {fmt(lw['pre_shots_against'])} to {fmt(lw['post_shots_against'])}{extra}")
            sub3_t.set_color(GREEN)

    # trace of the average pass position with substitution marks
    axt.set_facecolor(BG)
    if done:
        lw = done[-1]
        axt.axvspan(max(0, lw["t"] - W * 60), lw["t"], color="white", alpha=0.08, linewidth=0)
        axt.axvspan(lw["t"], min(T, lw["t"] + W * 60), color=ORANGE, alpha=0.14, linewidth=0)
    axt.plot(T_grid[: k + 1], f_roll[: k + 1], color=BLUE, linewidth=2.6)
    axt.plot(T_grid[: k + 1], o_roll[: k + 1], color=ORANGE, linewidth=2.6)
    for w in fw:
        if w["t"] <= T:
            axt.axvline(w["t"], color=BLUE, linewidth=1.4, linestyle="--")
            axt.plot([w["t"]], [y_lo + 1.5], marker="^", color=BLUE, markersize=9, zorder=5)
    for w in ow:
        if w["t"] <= T:
            axt.axvline(w["t"], color=ORANGE, linewidth=1.0, linestyle="--", alpha=0.7)
            axt.plot([w["t"]], [y_lo + 1.5], marker="^", color=ORANGE, markersize=8, zorder=5)
    cg_f = cg_o = 0
    for gt, gteam in goals:
        if gt <= T:
            if gteam == F:
                cg_f += 1
            else:
                cg_o += 1
            axt.axvline(gt, color="white", linewidth=1, linestyle=":")
            axt.text(gt, y_hi, f" {cg_f}-{cg_o}", color="white", fontsize=10, va="top", ha="left")
    axt.set_xlim(0, t_end)
    axt.set_ylim(y_lo, y_hi)
    axt.set_xticks(tick_t)
    axt.set_xticklabels(tick_l)
    axt.tick_params(colors="white", labelsize=9)
    for sp in axt.spines.values():
        sp.set_color(GRID)
    axt.set_ylabel("Average pass\nposition (yards)", color=MUTED, fontsize=9)
    axt.set_title("Triangles: substitution waves (blue = " + F + ", orange = " + O + ")",
                  color=MUTED, fontsize=9.5, pad=4)


frames = list(range(N_FRAMES)) + [N_FRAMES - 1] * HOLD_FRAMES
FPS = 20
RAW_VIDEO = "substitution_replay_raw.mp4"
FINAL_VIDEO = "substitution_replay.mp4"
anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames), interval=1000 / FPS)
writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
print(f"Rendering {len(frames)} frames ({len(frames) / FPS:.0f} seconds), this takes a minute...")
anim.save(RAW_VIDEO, writer=writer, dpi=108, savefig_kwargs={"facecolor": BG})
print("Saved replay:", RAW_VIDEO)


# ---------- append the charts to the video as end cards ----------
def append_end_cards(video_in, video_out, cards, seconds, size=1080, fps=20, bg="0d1b2a"):
    cards = [c for c in cards if os.path.exists(c)]
    if not cards:
        print("No chart PNGs found, keeping the replay without end cards.")
        os.replace(video_in, video_out)
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", video_in]
    for c in cards:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", c]
    norm = (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=0x{bg},"
            f"setsar=1,fps={fps},format=yuv420p")
    n = len(cards) + 1
    parts_f = [f"[{i}:v]{norm}[v{i}]" for i in range(n)]
    labels = "".join(f"[v{i}]" for i in range(n))
    fc = ";".join(parts_f) + f";{labels}concat=n={n}:v=1:a=0[out]"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", video_out]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:])
        print("End cards failed, keeping the replay without them.")
        os.replace(video_in, video_out)
        return
    os.remove(video_in)
    print("Added end cards:", ", ".join(cards))


append_end_cards(RAW_VIDEO, FINAL_VIDEO, END_CARDS, END_CARD_SECONDS, fps=FPS)
print("Saved video:", FINAL_VIDEO)