#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tools.calibrate_wall_pair_transform import read_visual_poses,robust_panel_transform

def pose_transform(a_path:Path,b_path:Path,min_inliers:int):
 a=read_visual_poses(a_path);b=read_visual_poses(b_path);frames=sorted(set(a)&set(b));
 if len(frames)<min_inliers:raise ValueError(f'need {min_inliers} overlapping A/B frames, found {len(frames)}')
 r,t,keep,audit=robust_panel_transform([a[x] for x in frames],[b[x] for x in frames],minimum_inliers=min_inliers,expected_wall_plane_angle_deg=45.0,wall_plane_angle_tolerance_deg=45.0)
 return r,t,frames,keep,audit

def difference(a_r,a_t,b_r,b_t):return float(np.linalg.norm(a_t-b_t)),float(np.degrees((a_r.inv()*b_r).magnitude()))
def load_layout(path):return json.loads(path.read_text())
def map_from_layout(layout,world_frame):return {'schema_version':'world-apriltag-map/1.0','map_id':layout['revision_id']+'-'+layout['board'],'world_frame':world_frame,'calibration_status':'VERIFIED_PRINT_LAYOUT','tag_outer_size_m':layout['tag_black_outer_size_mm']/1000,'tags':layout['tags']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--left-a-pose',type=Path,required=True);p.add_argument('--left-b-pose',type=Path,required=True);p.add_argument('--right-a-pose',type=Path,required=True);p.add_argument('--right-b-pose',type=Path,required=True);p.add_argument('--layout-a',type=Path,required=True);p.add_argument('--layout-b',type=Path,required=True);p.add_argument('--pair-id',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--minimum-inliers',type=int,default=20);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 la=load_layout(a.layout_a);lb=load_layout(a.layout_b);(a.output_dir/'panel_A_map.json').write_text(json.dumps(map_from_layout(la,'session_grid_A'),indent=2)+'\n');(a.output_dir/'panel_B_map.json').write_text(json.dumps(map_from_layout(lb,'session_grid_B'),indent=2)+'\n')
 lr,lt,lf,lk,laudit=pose_transform(a.left_a_pose,a.left_b_pose,a.minimum_inliers);rr,rt,rf,rk,raudit=pose_transform(a.right_a_pose,a.right_b_pose,a.minimum_inliers);pos,rot=difference(lr,lt,rr,rt)
 quaternions=np.vstack((lr.as_quat(),rr.as_quat()))
 if np.dot(quaternions[0],quaternions[1])<0:quaternions[1]*=-1
 fit_r=Rotation.from_quat(quaternions.mean(0));fit_t=(lt+rt)/2
 transformed=[]
 for tag in lb['tags']:
  item=dict(tag);item['panel']='grid_B';item['corners_m']=(fit_r.apply(np.asarray(tag['corners_m']))+fit_t).tolist();transformed.append(item)
 tags=[]
 for tag in la['tags']:
  item=dict(tag);item['panel']='grid_A';tags.append(item)
 tags.extend(transformed);gates={'cross_camera_translation_difference_m_at_most_0_010':pos<=.010,'cross_camera_rotation_difference_deg_at_most_0_5':rot<=.5,'left_inliers_at_least_minimum':int(lk.sum())>=a.minimum_inliers,'right_inliers_at_least_minimum':int(rk.sum())>=a.minimum_inliers,'left_position_residual_p95_m_at_most_0_040':laudit['position_residual_m']['p95']<=.040,'right_position_residual_p95_m_at_most_0_040':raudit['position_residual_m']['p95']<=.040};verified=all(gates.values());world={'schema_version':'world-apriltag-map/1.0','map_id':f'{a.pair_id}-a3-session-map','world_frame':'session_grid_A','physical_up_vector':[0,-1,0],'calibration_status':'VERIFIED_CAPTURE_CROSS_CAMERA_HOLDOUT' if verified else 'HOLDOUT_FAILED','tag_outer_size_m':.120,'expected_ids':[200,201,202,203,204,205,210,211,212,213,214,215],'panel_transform':{'parent_frame':'session_grid_A','child_frame':'session_grid_B','translation_m':fit_t.tolist(),'quaternion_xyzw':fit_r.as_quat().tolist(),'scale':1.0},'tags':tags};report={'schema_version':'capture-a3-panel-calibration/v1','pair_id':a.pair_id,'status':'VERIFIED' if verified else 'HOLDOUT_FAILED','cross_camera_transform_difference':{'translation_m':pos,'rotation_deg':rot},'left':{'overlap_frames':len(lf),'inliers':int(lk.sum()),'audit':laudit},'right':{'overlap_frames':len(rf),'inliers':int(rk.sum()),'audit':raudit},'gates':gates};(a.output_dir/'session_world_map.json').write_text(json.dumps(world,indent=2)+'\n');(a.output_dir/'session_world_map_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'translation_difference_mm':pos*1000,'rotation_difference_deg':rot,'gates':gates},indent=2));return 0 if verified else 2
if __name__=='__main__':raise SystemExit(main())
