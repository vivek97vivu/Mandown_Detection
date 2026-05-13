import cv2

from mmpose.apis import MMPoseInferencer

CONFIG = "/media/algosium/SSD/vivek/mandown_detection/test/rtmpose-m_8xb256-420e_coco-256x192.py"

CHECKPOINT = "/media/algosium/SSD/vivek/mandown_detection/models/pose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"

inferencer = MMPoseInferencer(
    pose2d=CONFIG,
    pose2d_weights=CHECKPOINT,
    device='cpu'
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    result_generator = inferencer(
        frame,
        return_vis=True,
        draw_bbox=True,
        show=False
    )

    result = next(result_generator)

    vis_frame = result['visualization'][0]

    cv2.imshow("RTMPose", vis_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()