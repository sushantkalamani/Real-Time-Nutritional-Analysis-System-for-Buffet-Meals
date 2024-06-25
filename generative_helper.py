import google.generativeai as genai
import os
from dotenv import load_dotenv
import cv2

def configure_api():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(image):
    input_prompt = '''You are an expert in nutritionist where you need to see the food items from the image, if you don't see any food plate and if the image is not stable just answer 'I don't see any plate', else
               calculate the total calories, also provide the details of every food items with calories intake
               is below format

               1. Item 1 - no of calories
               2. Item 2 - no of calories
               ----
               ----

        Finally you can also mention whether the food is healthy or not and also mention the percentage split of the ratio of carbohydrates, fats, fibers, sugar and other important things required in our diet
        '''
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input_prompt, image[0]])
    return response.text

def input_image_setup(frame):
    if frame is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        _, img_encoded = cv2.imencode('.jpg', rgb_frame)
        bytes_data = img_encoded.tobytes()
        
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise ValueError("No frame provided or frame is None")

if __name__ == "__main__":
    configure_api()


    image_file_path = "img_test.jpeg"
    
    try:
        image_data = input_image_setup(image_file_path)
        response = get_gemini_response(image_data)
        print(response)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
