"""
Hero figure for PPT: BVH skeleton gallery
- 7 emotions in one figure, each showing a representative skeleton pose
- Joints colored by per-joint average velocity (cold=slow, warm=fast)
- Bone thickness proportional to velocity
- Clean dark background suitable for slides
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path

import sys
sys.path.insert(0, str(Path("scripts/pipeline")))
from utils_bvh_parser import BVHParser

# ── config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("data/raw/kinematic-dataset-of-actors-expressing-emotions-2.1.0")
TEMPORAL_CSV = Path("outputs/analysis/temporal_3d/v1/bvh_temporal_features.csv")
OUT_DIR      = Path("docs/figs_3d_temporal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# representative file per emotion (median-velocity)
REP_FILES = {
    "Happy":    ("F08H3V2",  "F08"),
    "Angry":    ("F12A4V2",  "F12"),
    "Fearful":  ("F03F1V1",  "F03"),
    "Disgust":  ("F01D5V2",  "F01"),
    "Surprise": ("M03SU4V1", "M03"),
    "Neutral":  ("M10N3V1",  "M10"),
    "Sad":      ("F11SA5V1", "F11"),
}

EMO_ORDER = ["Happy", "Angry", "Fearful", "Disgust", "Surprise", "Neutral", "Sad"]

AVG_VEL = {  # from summary table
    "Happy":0.00824,"Angry":0.00600,"Fearful":0.00594,
    "Disgust":0.00425,"Surprise":0.00424,"Neutral":0.00364,"Sad":0.00343,
}

# skeleton connections (parent → child joint names in BVH)
BONES = [
    ("Hips","Spine"),("Spine","Spine1"),("Spine1","Spine2"),("Spine2","Spine3"),
    ("Spine3","Neck"),("Neck","Head"),
    ("Spine3","LeftShoulder"),("LeftShoulder","LeftArm"),
    ("LeftArm","LeftForeArm"),("LeftForeArm","LeftHand"),
    ("Spine3","RightShoulder"),("RightShoulder","RightArm"),
    ("RightArm","RightForeArm"),("RightForeArm","RightHand"),
    ("Hips","LeftUpLeg"),("LeftUpLeg","LeftLeg"),("LeftLeg","LeftFoot"),
    ("Hips","RightUpLeg"),("RightUpLeg","RightLeg"),("RightLeg","RightFoot"),
]

# joints we have velocity data for (from temporal analysis)
VEL_JOINT_MAP = {
    "Head":         "head_vel",
    "LeftArm":      "l_shoulder_vel",
    "RightArm":     "r_shoulder_vel",
    "LeftForeArm":  "l_elbow_vel",
    "RightForeArm": "r_elbow_vel",
    "LeftHand":     "l_wrist_vel",
    "RightHand":    "r_wrist_vel",
}

# load per-emotion per-joint velocity means
print("[+] Loading velocity data …")
tdf = pd.read_csv(TEMPORAL_CSV)
vel_means = {}
for emo in EMO_ORDER:
    sub = tdf[tdf["emotion"] == emo]
    vel_means[emo] = {j: sub[col].mean() for j, col in VEL_JOINT_MAP.items()}

# global velocity range for colormap
all_vels = [v for em in vel_means.values() for v in em.values()]
vmin, vmax = min(all_vels), max(all_vels)
cmap = cm.get_cmap("coolwarm")
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)


def get_joint_vel(emo, joint):
    return vel_means[emo].get(joint, (vmin + vmax) / 2)


def draw_skeleton(ax, coords, emo, show_ylabel=False):
    """Draw a single skeleton on ax. coords = {joint: [x,y,z]}"""
    # project to front view: X (left-right), Y (up-down)
    # center and scale
    all_pts = np.array(list(coords.values()))
    cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])

    # scale by body height
    y_range = all_pts[:, 1].max() - all_pts[:, 1].min()
    scale = 1.0 / (y_range + 1e-6)

    def pt(name):
        p = coords.get(name)
        if p is None:
            return None
        x = (p[0] - cx) * scale
        y = (p[1] - cy) * scale
        return x, y

    # draw bones
    for p_name, c_name in BONES:
        p_pt = pt(p_name)
        c_pt = pt(c_name)
        if p_pt is None or c_pt is None:
            continue
        # bone color = average of endpoint velocities
        v1 = get_joint_vel(emo, p_name)
        v2 = get_joint_vel(emo, c_name)
        bone_vel = (v1 + v2) / 2
        rgba = cmap(norm(bone_vel))
        lw = 1.5 + 4.0 * norm(bone_vel)
        ax.plot([p_pt[0], c_pt[0]], [p_pt[1], c_pt[1]],
                color=rgba, linewidth=lw, solid_capstyle="round",
                zorder=2, alpha=0.9)

    # draw joints
    for jname in coords:
        p = pt(jname)
        if p is None:
            continue
        v = get_joint_vel(emo, jname)
        rgba = cmap(norm(v))
        size = 20 + 120 * norm(v)
        ax.scatter(*p, color=rgba, s=size, zorder=3, edgecolors="white",
                   linewidths=0.4, alpha=0.95)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")


# ── find representative frame for each emotion ───────────────────────────────
def get_rep_frame(filename, actor, emo):
    """Return world coords of the frame with velocity closest to emotion median."""
    bvh_path = DATASET_ROOT / "BVH" / actor / f"{filename}.bvh"
    parser = BVHParser(str(bvh_path))
    n = len(parser.frames)

    # get a few frames and pick one near median activity
    # sample up to 30 evenly spaced frames
    idxs = list(range(0, n, max(1, n // 30)))
    frame_vels = []
    for i in idxs:
        c = parser.get_joint_world_coords(i)
        frame_vels.append((i, c))

    # pick frame near 75th percentile moment (avoid start/end neutral poses)
    target_idx = idxs[int(len(idxs) * 0.6)]
    return parser.get_joint_world_coords(target_idx)


# ── build figure ──────────────────────────────────────────────────────────────
print("[+] Building skeleton gallery …")

fig = plt.figure(figsize=(16, 5.5), facecolor="#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")

# emotion labels row + colorbar
n = len(EMO_ORDER)
axes = []
for i, emo in enumerate(EMO_ORDER):
    ax = fig.add_axes([0.02 + i * (0.94/n), 0.10, 0.94/n - 0.005, 0.78])
    ax.set_facecolor("#1a1a2e")
    axes.append(ax)

for i, emo in enumerate(EMO_ORDER):
    fname, actor = REP_FILES[emo]
    print(f"  loading {emo} ({fname}) …")
    coords = get_rep_frame(fname, actor, emo)
    draw_skeleton(axes[i], coords, emo)

    # emotion label
    avg_v = AVG_VEL[emo]
    color = cmap(norm(avg_v * 7))  # scale for visibility
    axes[i].set_title(emo, color="white", fontsize=13, fontweight="bold", pad=4)

    # velocity badge
    v_color = cmap(norm(avg_v * 7))
    axes[i].text(0.5, -0.05, f"avg_vel\n{avg_v:.5f}",
                 transform=axes[i].transAxes,
                 ha="center", va="top", color="white", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.25",
                           facecolor=cmap(norm(avg_v * 7)), alpha=0.6,
                           edgecolor="none"))

# colorbar
cax = fig.add_axes([0.25, 0.04, 0.50, 0.025])
sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("Normalised Joint Velocity  (cold = slow, warm = fast)",
             color="white", fontsize=9)
cb.ax.xaxis.set_tick_params(color="white", labelcolor="white")
plt.setp(cb.ax.get_xticklabels(), color="white")

# title
fig.text(0.5, 0.97, "3D Motion Capture – Skeleton Pose Gallery by Emotion",
         ha="center", va="top", color="white", fontsize=14, fontweight="bold")
fig.text(0.5, 0.93,
         "Joint colour & bone thickness = average velocity  |  Kinematic Dataset v2.1.0",
         ha="center", va="top", color="#aaaaaa", fontsize=9)

out_path = OUT_DIR / "3d_skeleton_gallery.png"
fig.savefig(out_path, dpi=200, facecolor="#1a1a2e", bbox_inches="tight")
plt.close(fig)
print(f"[✓] Saved → {out_path}")

# ── also produce a light-background version ───────────────────────────────────
print("[+] Building light-background version …")
cmap2 = cm.get_cmap("RdYlBu_r")

fig2 = plt.figure(figsize=(16, 5.5), facecolor="white")
axes2 = []
for i in range(n):
    ax = fig2.add_axes([0.02 + i * (0.94/n), 0.10, 0.94/n - 0.005, 0.78])
    ax.set_facecolor("#f5f5f5")
    axes2.append(ax)

for i, emo in enumerate(EMO_ORDER):
    fname, actor = REP_FILES[emo]
    coords = get_rep_frame(fname, actor, emo)

    all_pts = np.array(list(coords.values()))
    cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
    y_range = all_pts[:, 1].max() - all_pts[:, 1].min()
    scale = 1.0 / (y_range + 1e-6)

    def pt2(name):
        p = coords.get(name)
        if p is None: return None
        return (p[0]-cx)*scale, (p[1]-cy)*scale

    for p_name, c_name in BONES:
        p_pt = pt2(p_name); c_pt = pt2(c_name)
        if p_pt is None or c_pt is None: continue
        bone_vel = (get_joint_vel(emo, p_name)+get_joint_vel(emo, c_name))/2
        rgba = cmap2(norm(bone_vel))
        lw = 1.5 + 4.0*norm(bone_vel)
        axes2[i].plot([p_pt[0],c_pt[0]],[p_pt[1],c_pt[1]],
                      color=rgba, linewidth=lw, solid_capstyle="round", zorder=2)

    for jname in coords:
        p = pt2(jname)
        if p is None: continue
        v = get_joint_vel(emo, jname)
        rgba = cmap2(norm(v))
        axes2[i].scatter(*p, color=rgba, s=20+120*norm(v), zorder=3,
                         edgecolors="#333333", linewidths=0.4)

    axes2[i].set_xlim(-0.6,0.6); axes2[i].set_ylim(-0.7,0.7)
    axes2[i].set_aspect("equal"); axes2[i].axis("off")
    axes2[i].set_title(emo, fontsize=13, fontweight="bold", pad=4, color="#222222")
    avg_v = AVG_VEL[emo]
    axes2[i].text(0.5,-0.05,f"avg_vel={avg_v:.5f}",
                  transform=axes2[i].transAxes, ha="center", va="top",
                  fontsize=8, color="#444444")

cax2 = fig2.add_axes([0.25,0.04,0.50,0.025])
cb2 = fig2.colorbar(cm.ScalarMappable(cmap=cmap2,norm=mcolors.Normalize(vmin=vmin,vmax=vmax)),
                    cax=cax2, orientation="horizontal")
cb2.set_label("Normalised Joint Velocity  (blue=slow → red=fast)", fontsize=9)

fig2.text(0.5,0.97,"3D Motion Capture – Skeleton Pose Gallery by Emotion",
          ha="center",va="top",fontsize=14,fontweight="bold")
fig2.text(0.5,0.93,"Joint colour & thickness = average velocity  |  Kinematic Dataset v2.1.0",
          ha="center",va="top",fontsize=9,color="#666666")

out2 = OUT_DIR / "3d_skeleton_gallery_light.png"
fig2.savefig(out2, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig2)
print(f"[✓] Saved → {out2}")
