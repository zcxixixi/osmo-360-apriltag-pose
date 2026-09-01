#!/usr/bin/env python3
"""One wall anchor plus same-frame 20 mm BaseTag dual-camera localization."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.calibration.calibrate_basetag_reciprocal import Transform, rotation_distance_deg
from tools.joint_dual_camera_pose_graph_cached import (build_frames, direct_map,
    load_initial_wall_transform, raw_fisheye_cache_audit,
    solve_camera_wall_only, wall_support_score)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("left-cache","right-cache","left-initial-pose","right-initial-pose",
                 "left-panel-map","right-panel-map","initial-world-map"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--left-tag-id",type=int,required=True);p.add_argument("--right-tag-id",type=int,required=True)
    p.add_argument("--tag-size-m",type=float,default=.020);p.add_argument("--tag-corner-quarter-turns",type=int,default=1)
    p.add_argument("--left-tag-corner-quarter-turns",type=int,choices=range(4));p.add_argument("--right-tag-corner-quarter-turns",type=int,choices=range(4))
    p.add_argument("--start-common-s",type=float,required=True);p.add_argument("--end-common-s",type=float,required=True)
    p.add_argument("--sample-stride",type=int,default=1)
    p.add_argument("--max-cross-basetag-center-error-deg",type=float,default=50.)
    p.add_argument("--max-cross-basetag-attitude-error-deg",type=float,default=20.)
    p.add_argument("--left-camera-to-tag",type=Path)
    p.add_argument("--right-camera-to-tag",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    return p.parse_args()


def choose_episode_anchor(frames):
    scores={}
    for side,index,cross in (("left",0,"cross_lr_pose"),("right",1,"cross_rl_pose")):
        both=sum(wall_support_score(f,index)[0] for f in frames);tags=sum(wall_support_score(f,index)[1] for f in frames)
        usable=sum(getattr(f,cross) is not None for f in frames)
        scores[side]={"both_wall_frames":both,"wall_tag_observations":tags,"usable_cross_frames":usable,
                      "rank":[min(both,usable),usable,both,tags]}
    return max(scores,key=lambda s:tuple(scores[s]["rank"])),scores


def direct_camera_pair(frame,anchor_side,wall,own_left,own_right,previous_anchor=None):
    if anchor_side=="right":
        if frame.cross_rl_pose is None:return None
        right=solve_camera_wall_only(previous_anchor or frame.initial_right,frame.right_leftwall,frame.right_rightwall,wall)
        return right.compose(frame.cross_rl_pose).compose(own_left.inverse()),right
    if frame.cross_lr_pose is None:return None
    left=solve_camera_wall_only(previous_anchor or frame.initial_left,frame.left_leftwall,frame.left_rightwall,wall)
    return left,left.compose(frame.cross_lr_pose).compose(own_right.inverse())


def stats(values):
    if not values:return {"count":0}
    x=np.asarray(values,float)
    return {"count":len(x),"min":float(x.min()),"median":float(np.median(x)),"p95":float(np.percentile(x,95)),"max":float(x.max())}


def write_pose(path,rows,side):
    fields=["frame","timestamp","camera_x_m","camera_y_m","camera_z_m","qx","qy","qz","qw","parent_frame","child_frame","measurement_source","quality_status","anchor_side"]
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            T=row[side];q=T.r.as_quat();w.writerow({"frame":round(row["t"]*30),"timestamp":f'{row["t"]:.9f}',"camera_x_m":T.p[0],"camera_y_m":T.p[1],"camera_z_m":T.p[2],"qx":q[0],"qy":q[1],"qz":q[2],"qw":q[3],"parent_frame":"tag_map","child_frame":f"{side}_fisheye1_camera","measurement_source":"direct_reciprocal_raw_fisheye","quality_status":"measured","anchor_side":row["anchor"]})


def main():
    a=parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);a.temporal_blocks=5
    inputs={"left":raw_fisheye_cache_audit(a.left_cache),"right":raw_fisheye_cache_audit(a.right_cache)}
    h=a.tag_size_m/2;tag=np.array([[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0.]])
    frames,ol,orr,audit=build_frames(a,direct_map(a.left_panel_map),direct_map(a.right_panel_map),tag)
    anchor,scores=choose_episode_anchor(frames);wall=load_initial_wall_transform(a.initial_world_map)
    tb=Transform(np.array([.02625,0,.0196]),Rotation.identity()).inverse();rows=[];prev=None;prev_t=None;sep=[];change=[];hp=[];hr=[]
    for frame in sorted(frames,key=lambda f:f.time_s):
        prior=prev if prev_t is not None and frame.time_s-prev_t<=.15 else None
        pair=direct_camera_pair(frame,anchor,wall,ol,orr,prior)
        if pair is None:continue
        left,right=pair;prev=right if anchor=="right" else left;prev_t=frame.time_s
        lb=left.compose(ol).compose(tb);rb=right.compose(orr).compose(tb);d=float(np.linalg.norm(lb.p-rb.p));sep.append(d)
        if rows and frame.time_s-rows[-1]["t"]<=.15:change.append(abs(d-rows[-1]["separation_m"]))
        observed=frame.cross_lr_pose if anchor=="right" else frame.cross_rl_pose
        predicted=left.inverse().compose(right.compose(orr)) if anchor=="right" else right.inverse().compose(left.compose(ol))
        if observed is not None:hp.append(float(np.linalg.norm(observed.p-predicted.p)*1000));hr.append(rotation_distance_deg(observed.r,predicted.r))
        rows.append({"t":frame.time_s,"left":left,"right":right,"anchor":anchor,"separation_m":d})
    write_pose(a.output_dir/"left_pose.csv",rows,"left");write_pose(a.output_dir/"right_pose.csv",rows,"right")
    hp_stats=stats(hp);hr_stats=stats(hr);continuity=stats(change)
    gates={"holdout_count_min":20,"position_median_mm_max":3.,"position_p95_mm_max":5.,"rotation_median_deg_max":2.,"rotation_p95_deg_max":3.,"adjacent_separation_change_p95_m_max":.020}
    passed=(hp_stats["count"]>=gates["holdout_count_min"] and hp_stats.get("median",1e9)<=3 and hp_stats.get("p95",1e9)<=5 and hr_stats.get("median",1e9)<=2 and hr_stats.get("p95",1e9)<=3 and continuity.get("p95",1e9)<=.020)
    report={"schema_version":"direct-reciprocal-world-pose/1.0","status":"PASS" if passed else "HOLDOUT_FAILED","algorithm":"single episode-level wall anchor + same-frame reciprocal BaseTag","anchor_side":anchor,"anchor_scores":scores,"frame_count":len(rows),"wall_tag_size_m":.200,"basetag_size_m":a.tag_size_m,"base_separation_m":stats(sep),"adjacent_base_separation_change_m":continuity,"opposite_direction_holdout_position_mm":hp_stats,"opposite_direction_holdout_rotation_deg":hr_stats,"gates":gates,"own_basetag":{k:audit[k] for k in ("left","right")},"cross_selection":audit["cross_basetag_selection"],"metric_input_audit":inputs,"contact_constraint_used":False,"stitched_input_used":False,"training_ready":passed}
    (a.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
