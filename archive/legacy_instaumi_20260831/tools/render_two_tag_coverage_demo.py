#!/usr/bin/env python3
"""Render a fast coverage audit when a two-Tag capture cannot produce a trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--lens0-detections", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args();args.output_dir.mkdir(parents=True, exist_ok=False)
    samples = json.loads(args.detections.read_text(encoding="utf-8"))
    lens0 = json.loads(args.lens0_detections.read_text(encoding="utf-8")) if args.lens0_detections else []
    states = []
    for sample in samples:
        ids = set(map(int, sample["detections"]))
        states.append(2 if ids == {200, 201} else 1 if ids else 0)
    count = len(states);both=states.count(2);one=states.count(1);none=states.count(0)
    lens0_any = sum(bool(sample["detections"]) for sample in lens0)
    report = {
        "schema_version": "two-screen-tag-coverage/1.0",
        "status": "FAIL_INSUFFICIENT_TWO_TAG_COVERAGE_FOR_TRAJECTORY",
        "samples": count,
        "lens1": {"both": both, "both_ratio": both/count, "one": one, "one_ratio": one/count, "none": none, "none_ratio": none/count},
        "lens0": {"samples": len(lens0), "any_tag": lens0_any, "any_tag_ratio": lens0_any/len(lens0) if lens0 else None},
        "trajectory_generated": False,
        "reason": "Only three 10 Hz samples contain both IDs; no reliable two-Tag map or continuous frozen-locator trajectory can be established.",
    }
    (args.output_dir/"coverage_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    capture=cv2.VideoCapture(str(args.video));source_fps=float(capture.get(cv2.CAP_PROP_FPS));frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));output_fps=30.0;output_frames=round(frame_count/source_fps*output_fps)
    writer=cv2.VideoWriter(str(args.output_dir/"two_tag_coverage_failure_demo.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),output_fps,(1280,720))
    colors={200:(60,220,255),201:(100,230,120)}
    current_source_frame=-1
    frame=None
    for index in range(output_frames):
        time_s=index/output_fps;source_frame=min(frame_count-1,round(time_s*source_fps));ok=True
        while current_source_frame<source_frame:
            ok,frame=capture.read();current_source_frame+=1
            if not ok:break
        if not ok or frame is None:break
        view=cv2.resize(frame,(720,720),interpolation=cv2.INTER_AREA);sample=min(samples,key=lambda item:abs(item["time_s"]-time_s));ids=sorted(map(int,sample["detections"]))
        for tag_id_text,quad in sample["detections"].items():
            tag_id=int(tag_id_text);points=np.round(np.asarray(quad)*720/1920).astype(np.int32);cv2.polylines(view,[points],True,colors[tag_id],3,cv2.LINE_AA);cv2.putText(view,f"ID {tag_id}",tuple(points[0]),cv2.FONT_HERSHEY_SIMPLEX,.55,colors[tag_id],2)
        canvas=np.full((720,1280,3),(12,18,25),np.uint8);canvas[:,:720]=view
        cv2.putText(canvas,"REAL 0065 / TWO SCREEN TAG COVERAGE",(742,45),cv2.FONT_HERSHEY_SIMPLEX,.70,(80,215,245),2)
        cv2.putText(canvas,"FROZEN LOCATOR: TRAJECTORY NOT GENERATED",(742,82),cv2.FONT_HERSHEY_SIMPLEX,.54,(80,120,255),2)
        cv2.putText(canvas,f"current IDs  {ids if ids else 'NONE'}",(742,135),cv2.FONT_HERSHEY_SIMPLEX,.62,(220,225,232),2)
        cv2.putText(canvas,f"both Tags  {both}/{count}  ({both/count*100:.1f}%)",(742,185),cv2.FONT_HERSHEY_SIMPLEX,.65,(100,230,120),2)
        cv2.putText(canvas,f"one Tag    {one}/{count}  ({one/count*100:.1f}%)",(742,225),cv2.FONT_HERSHEY_SIMPLEX,.60,(230,190,80),2)
        cv2.putText(canvas,f"no Tag     {none}/{count}  ({none/count*100:.1f}%)",(742,265),cv2.FONT_HERSHEY_SIMPLEX,.60,(90,120,255),2)
        cv2.rectangle(canvas,(742,320),(1240,560),(55,70,82),1)
        for sample_index,state in enumerate(states):
            x0=750+round(sample_index*480/count);x1=750+round((sample_index+1)*480/count);color=(80,210,110) if state==2 else (60,180,230) if state==1 else (65,75,88);cv2.rectangle(canvas,(x0,390),(max(x0+1,x1),485),color,-1)
        marker=min(count-1,round(time_s/(frame_count/source_fps)*count));mx=750+round(marker*480/count);cv2.line(canvas,(mx,370),(mx,510),(245,245,245),2)
        cv2.putText(canvas,"gray = none   amber = one   green = both",(760,350),cv2.FONT_HERSHEY_SIMPLEX,.50,(190,200,210),1)
        cv2.putText(canvas,"RECORD AGAIN: KEEP BOTH TAGS FULLY VISIBLE",(742,630),cv2.FONT_HERSHEY_SIMPLEX,.58,(80,120,255),2)
        cv2.putText(canvas,"No absolute or trajectory claim is made from this capture.",(742,675),cv2.FONT_HERSHEY_SIMPLEX,.44,(105,120,132),1)
        writer.write(canvas)
    capture.release();writer.release();print(json.dumps(report,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
