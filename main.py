import argparse
import mediapipe as mp
import cv2


def face_detection() -> None:
    # ref: https://google.github.io/mediapipe/solutions/face_detection.html#overview

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # なにこれ？
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)  # 接続されているデバイスの0番目．認識されているとは限らない
    with mp_face_detection.FaceDetection(  # withで開くことで，close処理を忘れない
        model_selection=1,  # 0:short-range(within 2[m]) / 1:full-range(within 5[m])
        min_detection_confidence=0.5  # ゼミでやったconfidence値
    ) as fd:
        while cap.isOpened():
            status, image = cap.read()
            if not status:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ConVerT color (bgr -> rgb)
            results = fd.process(image)  # main process
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # re ConVerT color (bgr -> rgb)

            # Draw the face detection annotations on the image.
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

            # ウィンドウサイズを可変にする．第一引数のidはimshowと一致させる必要がある(飾りでは無い)
            cv2.namedWindow("MediaPipe Face Detection", cv2.WINDOW_NORMAL)
            # eng:Flip the image horizontally for a selfie-view display.
            # ja:cv2.flip(image, 1)で左右反転して鏡の様にしている
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            key = cv2.waitKey(5)  # 5[msec]の間入力待機
            if key == 27:  # key == [ESC]
                break
    # close処理
    cap.release()
    return


def face_mesh() -> None:
    # ref: https://google.github.io/mediapipe/solutions/face_mesh.html#overview

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:
        while cap.isOpened():
            status, image = cap.read()
            if not status:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = fm.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw the face mesh annotations on the image.
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                    )

            cv2.namedWindow("MediaPipe Face Mesh", cv2.WINDOW_NORMAL)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            key = cv2.waitKey(5)
            if key == 27:
                break
    cap.release()
    return


def hands() -> None:
    # ref: https://google.github.io/mediapipe/solutions/hands.html#overview

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hd:
        while cap.isOpened():
            status, image = cap.read()
            if not status:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hd.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw the hand annotations on the image.
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            cv2.namedWindow("MediaPipe Hands", cv2.WINDOW_NORMAL)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            key = cv2.waitKey(5)
            if key == 27:
                break
    cap.release()
    return


def holistic() -> None:
    # ref: https://google.github.io/mediapipe/solutions/holistic.html#overview

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hs:
        while cap.isOpened():
            status, image = cap.read()
            if not status:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hs.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmark annotation on the image.
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())

            cv2.namedWindow("MediaPipe Holistic", cv2.WINDOW_NORMAL)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            key = cv2.waitKey(5)
            if key == 27:
                break
    cap.release()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Media Pipe test")
    parser.add_argument(
        "--mode",
        type=str,
        default="face_detection",
        choices=[
            "face_detection",
            "face_mesh",
            "hands",
            "holistic",
        ],
        help="select mode",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "face_detection":
        face_detection()
    elif mode == "face_mesh":
        face_mesh()
    elif mode == "hands":
        hands()
    elif mode == "holistic":
        holistic()

    cv2.destroyAllWindows()
