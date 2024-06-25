import cv2
import time
import generative_helper as gh

def main():
    gh.configure_api()

    video_path = "v1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_rate = 1  
    prev = 0

    # width and height 
    desired_width = 640
    desired_height = 480

    while cap.isOpened():
        time_elapsed = time.time() - prev

        ret, frame = cap.read()
        if not ret:
            break

        if time_elapsed > frame_rate:
            prev = time.time()

            try:
                resized_frame = cv2.resize(frame, (desired_width, desired_height))

                image_data = gh.input_image_setup(resized_frame)
                response = gh.get_gemini_response(image_data)
                print(response)
                
                cv2.imshow('Video Frame', resized_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"An error occurred in main: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
