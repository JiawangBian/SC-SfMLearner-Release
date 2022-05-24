TEST_DRIVE_ID='2021-08-13--09-27-11, 2021-09-10--13-58-21'
CAMERA_VIEW='cam_front_center'
python test_iterable_dataset.py \
--dataset-path '/nas/drives/yaak/yaak_dataset/video_data' \
--cam-calib-path '/nas/drives/yaak/yaak_dataset/camera_calibration' \
--test-drive-id "$TEST_DRIVE_ID" \
--camera-view "$CAMERA_VIEW" \
--video-format 'mp4' \
--telemetry-filename 'metadata.json' \
--batch-size 4 \
--video-clip-length 3 \
--video-clip-step 30 \
--frame-target-size '-1, -1' \
--max-iter 5 \
--device-id 0 \
--use-gpu \
--return-camera-intrinsics \
--return-tgt-ref-frames \
--return-processed-videos \
--return-telemetry
