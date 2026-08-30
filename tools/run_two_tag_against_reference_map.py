#!/usr/bin/env python3
"""Validate screen Tags 200/201 against the verified room wall-Tag trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from osmo360.localization.raw_fisheye_world_pose import make_ray_converter
from tools.run_physical_two_tag_camera_experiment import (
    compose, inverse, local_corners, pose_from_tag, stats,
)
from tools.run_two_tag_synthetic_experiment import verify_freeze
from osmo360.localization.world_frames import compile_world_tag_map


from tools._root import ROOT
PANO_ROOT = ROOT.parent / "panoforge-test"
SCREEN_IDS = (200, 201)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stream", type=int, default=1)
    parser.add_argument("--capture-id")
    return parser.parse_args()


def load_cache(path: Path):
    cache = np.load(path)
    by_frame = defaultdict(list)
    for index, frame in enumerate(cache["frame_index"]):
        by_frame[int(frame)].append(index)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return cache, by_frame, metadata


def tag_frame(corners: np.ndarray) -> tuple[np.ndarray, Rotation, float]:
    center = corners.mean(axis=0)
    x_axis = corners[1] - corners[0];size = float(np.linalg.norm(x_axis));x_axis /= size
    y_axis = corners[3] - corners[0];y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis);z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return center, Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis))), size


def write_pose(path: Path, rows: list[tuple[int, float, np.ndarray, Rotation]]) -> None:
    fields = ["frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m", "qx", "qy", "qz", "qw"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields);writer.writeheader()
        for frame, time_s, position, rotation in rows:
            quaternion = rotation.as_quat()
            writer.writerow({"frame":frame,"timestamp":f"{time_s:.6f}","camera_x_m":position[0],"camera_y_m":position[1],"camera_z_m":position[2],"qx":quaternion[0],"qy":quaternion[1],"qz":quaternion[2],"qw":quaternion[3]})


def build_reference_initial(cache, by_frame, metadata, world_map: dict, convert) -> list[tuple]:
    map_corners = {int(tag["id"]):np.asarray(tag["corners_m"],float) for tag in world_map["tags"]}
    rows=[]
    for frame in sorted(by_frame):
        candidates=[]
        for index in by_frame[frame]:
            tag_id=int(cache["tag_id"][index])
            if tag_id not in map_corners:continue
            world_center, world_rotation, size = tag_frame(map_corners[tag_id])
            try:tag_position,tag_rotation,rmse=pose_from_tag(local_corners(size),cache["corners_px"][index],convert)
            except ValueError:continue
            world_camera=compose((world_center,world_rotation),(tag_position,tag_rotation))
            candidates.append((float(cache["area_px2"][index]),rmse,world_camera))
        if candidates:
            _,_,(position,rotation)=max(candidates,key=lambda item:item[0])
            rows.append((frame,frame/float(metadata["fps"]),position,rotation))
    return rows


def run_cached(cache_path: Path, tag_map: Path, initial: Path, output: Path, min_tags: int, stride: int, regularize: bool) -> None:
    command=[str(ROOT/".venv/bin/python"),"-m","tools.raw_fisheye_world_pose_cached","--observation-cache",str(cache_path),"--tag-map",str(tag_map),"--initial-pose",str(initial),"--sample-stride",str(stride),"--min-tags",str(min_tags),"--max-angular-rmse-deg","1.5","--prior-policy","initial-first","--output-dir",str(output)]
    if regularize:command.append("--regularize-prior")
    subprocess.run(command,check=True)


def read_pose(path: Path):
    rows=[row for row in csv.DictReader(path.open(newline="",encoding="utf-8")) if row.get("camera_x_m")]
    result={}
    for row in rows:
        frame=int(row["frame"]);position=np.asarray([float(row[key]) for key in ("camera_x_m","camera_y_m","camera_z_m")]);rotation=Rotation.from_quat([float(row[key]) for key in ("qx","qy","qz","qw")]);result[frame]=(float(row["timestamp"]),position,rotation)
    return result


def fit_screen_tag(tag_id: int, observations: list[dict], reference: dict, convert):
    usable=[]
    for observation in observations:
        frame=observation["frame"]
        if frame not in reference:continue
        _,camera_position,camera_rotation=reference[frame]
        rays=convert(observation["quad"])
        usable.append((frame,camera_position,camera_rotation,rays,observation["quad"]))
    if len(usable)<20:raise RuntimeError(f"ID{tag_id} has only {len(usable)} reference-aligned observations")
    frame,camera_position,camera_rotation,_,quad=usable[len(usable)//2]
    tag_position,tag_rotation,_=pose_from_tag(local_corners(.24),quad,convert)
    world_tag=compose((camera_position,camera_rotation),inverse((tag_position,tag_rotation)))
    x0=np.r_[world_tag[0],world_tag[1].as_rotvec(),np.log(.24)]
    calibration=[item for index,item in enumerate(usable) if index%5!=0];holdout=[item for index,item in enumerate(usable) if index%5==0]
    unit=local_corners(1.0)
    def residual(parameters,data):
        center=parameters[:3];rotation=Rotation.from_rotvec(parameters[3:6]);size=float(np.exp(parameters[6]));world=rotation.apply(unit*size)+center;parts=[]
        for _,camera_p,camera_r,rays,_ in data:
            predicted=camera_r.inv().apply(world-camera_p);predicted/=np.linalg.norm(predicted,axis=1,keepdims=True);parts.append((predicted-rays).ravel())
        return np.concatenate(parts)
    fit=least_squares(lambda value:residual(value,calibration),x0,loss="huber",f_scale=.003,max_nfev=3000)
    center=fit.x[:3];rotation=Rotation.from_rotvec(fit.x[3:6]);size=float(np.exp(fit.x[6]));corners=rotation.apply(unit*size)+center
    holdout_vector=residual(fit.x,holdout).reshape(-1,3);holdout_angle=np.degrees(np.arccos(np.clip(1-np.sum(holdout_vector**2,axis=1)/2,-1,1)))
    return {"id":tag_id,"corners_m":corners.tolist(),"panel":f"screen_id{tag_id}"},{"observations":len(usable),"calibration_observations":len(calibration),"holdout_observations":len(holdout),"fitted_size_m":size,"holdout_angular_error_deg":stats(holdout_angle)}


def render_demo(output: Path, video: Path, cache, screen_pose: dict, reference: dict, report: dict, fps: float, frame_count: int, capture_id: str) -> None:
    capture=cv2.VideoCapture(str(video));output_fps=30.0;output_frames=round(frame_count/fps*output_fps);writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),output_fps,(1280,720));by_frame=defaultdict(list)
    for index,frame in enumerate(cache["frame_index"]):
        if int(cache["tag_id"][index]) in SCREEN_IDS:by_frame[int(frame)].append(index)
    common=sorted(set(screen_pose)&set(reference));ref=np.asarray([reference[f][1] for f in common]);pred=np.asarray([screen_pose[f][1] for f in common]);xy=np.vstack((ref[:,[0,2]],pred[:,[0,2]]));low=xy.min(0);span=np.maximum(xy.max(0)-low,1e-4);current=-1;image=None
    def points(values):
        normalized=(values[:,[0,2]]-low)/span;return np.c_[770+normalized[:,0]*430,610-normalized[:,1]*320].astype(np.int32)
    for out_index in range(output_frames):
        time_s=out_index/output_fps;target=min(frame_count-1,round(time_s*fps));ok=True
        while current<target:ok,image=capture.read();current+=1
        if not ok or image is None:break
        view=cv2.resize(image,(720,720));nearest=min(common,key=lambda frame:abs(frame-target));indices=by_frame.get(min(by_frame,key=lambda frame:abs(frame-target)),[])
        for index in indices:
            tag_id=int(cache["tag_id"][index])
            source_height,source_width=image.shape[:2]
            scale=np.asarray([720/source_width,720/source_height])
            quad=np.round(cache["corners_px"][index]*scale).astype(np.int32)
            color=(60,220,255) if tag_id==200 else (100,230,120)
            cv2.polylines(view,[quad],True,color,3)
            cv2.putText(view,f"ID {tag_id}",tuple(quad[0]),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        canvas=np.full((720,1280,3),(12,18,25),np.uint8)
        canvas[:,:720]=view
        cv2.putText(canvas,f"REAL {capture_id.upper()} / TWO SCREEN TAG TRAJECTORY",(742,44),cv2.FONT_HERSHEY_SIMPLEX,.52,(80,215,245),2)
        cv2.putText(canvas,f"reference-aligned screen pose frames {len(common)}",(742,82),cv2.FONT_HERSHEY_SIMPLEX,.48,(210,220,230),1)
        cv2.putText(canvas,f"position P95 {report['screen_vs_reference']['position_error_mm']['p95']:.2f} mm",(742,135),cv2.FONT_HERSHEY_SIMPLEX,.65,(120,230,145),2)
        cv2.putText(canvas,f"orientation P95 {report['screen_vs_reference']['orientation_error_deg']['p95']:.2f} deg",(742,175),cv2.FONT_HERSHEY_SIMPLEX,.58,(120,230,145),2)
        cv2.rectangle(canvas,(742,230),(1240,650),(55,70,82),1)
        subset=[frame for frame in common if frame<=nearest]
        cv2.polylines(canvas,[points(np.asarray([reference[f][1] for f in subset]))],False,(60,190,255),4)
        cv2.polylines(canvas,[points(np.asarray([screen_pose[f][1] for f in subset]))],False,(245,245,245),1)
        cv2.putText(canvas,"orange=verified wall reference  white=screen Tags",(760,265),cv2.FONT_HERSHEY_SIMPLEX,.46,(200,210,220),1)
        cv2.putText(canvas,"SCREEN MAP CALIBRATED AGAINST VERIFIED WALL MAP",(744,690),cv2.FONT_HERSHEY_SIMPLEX,.42,(105,120,132),1)
        writer.write(canvas)
    capture.release();writer.release()


def main() -> int:
    args=parse_args();verify_freeze();args.output_dir.mkdir(parents=True,exist_ok=False);cache,by_frame,metadata=load_cache(args.observation_cache)
    source_width,source_height=metadata["source_size"]
    geometry=SimpleNamespace(calibration=args.calibration.resolve(),panoforge_root=(ROOT.parent/"panoforge-test").resolve(),source_width=int(source_width),source_height=int(source_height),stream=args.stream,radial_model="stitch");convert,_=make_ray_converter(geometry)
    reference_map=compile_world_tag_map(args.reference_map);reference_initial=build_reference_initial(cache,by_frame,metadata,reference_map,convert);initial_path=args.output_dir/"reference_initial_pose.csv";write_pose(initial_path,reference_initial);compiled_reference=args.output_dir/"reference_map_snapshot.json";compiled_reference.write_text(json.dumps(reference_map,indent=2)+"\n")
    reference_dir=args.output_dir/"reference-locator";stride=int(metadata["frame_stride"]);run_cached(args.observation_cache,args.reference_map.resolve(),initial_path,reference_dir,2,stride,False);reference=read_pose(reference_dir/"pose.csv")
    observations={tag_id:[] for tag_id in SCREEN_IDS}
    for frame,indices in by_frame.items():
        for index in indices:
            tag_id=int(cache["tag_id"][index])
            if tag_id in SCREEN_IDS:observations[tag_id].append({"frame":frame,"quad":np.asarray(cache["corners_px"][index])})
    screen_tags=[];screen_audit={}
    for tag_id in SCREEN_IDS:
        tag,audit=fit_screen_tag(tag_id,observations[tag_id],reference,convert);screen_tags.append(tag);screen_audit[str(tag_id)]=audit
    capture_id=args.capture_id or args.video.stem
    screen_map={"schema_version":"world-apriltag-map/1.0","map_id":f"{capture_id}-screen-tags-calibrated-against-verified-room-v1","calibration_status":"CALIBRATED_AGAINST_VERIFIED_ROOM_TAG_MAP_SAME_CAPTURE","world_frame":reference_map["world_frame"],"physical_up_vector":reference_map.get("physical_up_vector",[0,-1,0]),"units":"m","expected_ids":[200,201],"tags":screen_tags};screen_map_path=args.output_dir/"screen_tag_map.json";screen_map_path.write_text(json.dumps(screen_map,indent=2)+"\n")
    screen_dir=args.output_dir/"screen-locator";run_cached(args.observation_cache,screen_map_path,initial_path,screen_dir,1,stride,False);screen_pose=read_pose(screen_dir/"pose.csv");common=sorted(set(screen_pose)&set(reference));position_error=np.asarray([np.linalg.norm(screen_pose[f][1]-reference[f][1])*1000 for f in common]);orientation_error=np.asarray([np.degrees((reference[f][2].inv()*screen_pose[f][2]).magnitude()) for f in common]);summary=json.loads((screen_dir/"summary.json").read_text())
    report={
        "schema_version":"two-screen-vs-verified-room/1.0",
        "capture_id":capture_id,
        "status":"DIAGNOSTIC",
        "absolute_reference":"verified room 10-Tag map",
        "reference_valid_frames":len(reference),
        "screen_valid_frames":len(screen_pose),
        "common_frames":len(common),
        "screen_locator_valid_ratio":summary["valid_ratio"],
        "screen_angular_rmse_deg":summary["angular_rmse_deg"],
        "screen_tag_calibration":screen_audit,
        "screen_vs_reference":{
            "position_error_mm":stats(position_error),
            "orientation_error_deg":stats(orientation_error),
        },
        "limitations":[
            "screen Tag map and evaluation use disjoint observations from the same capture, not a separate capture",
            "monitor physical size is inferred through the verified room reference",
            "not training-ready ground truth",
        ],
    }
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    render_demo(args.output_dir/f"{capture_id}_screen_tag_trajectory_demo.mp4",args.video,cache,screen_pose,reference,report,float(metadata["fps"]),int(metadata["frame_count"]),capture_id)
    cache.close()
    print(json.dumps(report,indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
