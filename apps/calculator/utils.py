# # import torch
# # from transformers import pipeline, BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration
# # from PIL import Image

# # # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
# # quantization_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_compute_dtype=torch.float16
# # )


# # model_id = "llava-hf/llava-1.5-7b-hf"
# # processor = AutoProcessor.from_pretrained(model_id)
# # model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
# # # pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

# # def analyze_image(image: Image):
# #     prompt = "USER: <image>\nAnalyze the equation or expression in this image, and return answer in format: {expr: given equation in LaTeX format, result: calculated answer}"

# #     inputs = processor(prompt, images=[image], padding=True, return_tensors="pt").to("cuda")
# #     for k, v in inputs.items():
# #         print(k,v.shape)

# #     output = model.generate(**inputs, max_new_tokens=20)
# #     generated_text = processor.batch_decode(output, skip_special_tokens=True)
# #     for text in generated_text:
# #         print(text.split("ASSISTANT:")[-1])

# import google.generativeai as genai
# import ast
# import json
# from PIL import Image
# from constants import GEMINI_API_KEY

# genai.configure(api_key=GEMINI_API_KEY)

# def analyze_image(img: Image, dict_of_vars: dict):
#     model = genai.GenerativeModel(model_name="gemini-1.5-flash")
#     dict_of_vars_str = json.dumps(dict_of_vars, ensure_ascii=False)
#     prompt = (
#         f"You have been given an image with some mathematical expressions, equations, or graphical problems, and you need to solve them. "
#         f"Note: Use the PEMDAS rule for solving mathematical expressions. PEMDAS stands for the Priority Order: Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right). Parentheses have the highest priority, followed by Exponents, then Multiplication and Division, and lastly Addition and Subtraction. "
#         f"For example: "
#         f"Q. 2 + 3 * 4 "
#         f"(3 * 4) => 12, 2 + 12 = 14. "
#         f"Q. 2 + 3 + 5 * 4 - 8 / 2 "
#         f"5 * 4 => 20, 8 / 2 => 4, 2 + 3 => 5, 5 + 20 => 25, 25 - 4 => 21. "
#         f"YOU CAN HAVE FIVE TYPES OF EQUATIONS/EXPRESSIONS IN THIS IMAGE, AND ONLY ONE CASE SHALL APPLY EVERY TIME: "
#         f"Following are the cases: "
#         f"1. Simple mathematical expressions like 2 + 2, 3 * 4, 5 / 6, 7 - 8, etc.: In this case, solve and return the answer in the format of a LIST OF ONE DICT [{{'expr': given expression, 'result': calculated answer}}]. "
#         f"2. Set of Equations like x^2 + 2x + 1 = 0, 3y + 4x = 0, 5x^2 + 6y + 7 = 12, etc.: In this case, solve for the given variable, and the format should be a COMMA SEPARATED LIST OF DICTS, with dict 1 as {{'expr': 'x', 'result': 2, 'assign': True}} and dict 2 as {{'expr': 'y', 'result': 5, 'assign': True}}. This example assumes x was calculated as 2, and y as 5. Include as many dicts as there are variables. "
#         f"3. Assigning values to variables like x = 4, y = 5, z = 6, etc.: In this case, assign values to variables and return another key in the dict called {{'assign': True}}, keeping the variable as 'expr' and the value as 'result' in the original dictionary. RETURN AS A LIST OF DICTS. "
#         f"4. Analyzing Graphical Math problems, which are word problems represented in drawing form, such as cars colliding, trigonometric problems, problems on the Pythagorean theorem, adding runs from a cricket wagon wheel, etc. These will have a drawing representing some scenario and accompanying information with the image. PAY CLOSE ATTENTION TO DIFFERENT COLORS FOR THESE PROBLEMS. You need to return the answer in the format of a LIST OF ONE DICT [{{'expr': given expression, 'result': calculated answer}}]. "
#         f"5. Detecting Abstract Concepts that a drawing might show, such as love, hate, jealousy, patriotism, or a historic reference to war, invention, discovery, quote, etc. USE THE SAME FORMAT AS OTHERS TO RETURN THE ANSWER, where 'expr' will be the explanation of the drawing, and 'result' will be the abstract concept. "
#         f"Analyze the equation or expression in this image and return the answer according to the given rules: "
#         f"Make sure to use extra backslashes for escape characters like \\f -> \\\\f, \\n -> \\\\n, etc. "
#         f"Here is a dictionary of user-assigned variables. If the given expression has any of these variables, use its actual value from this dictionary accordingly: {dict_of_vars_str}. "
#         f"DO NOT USE BACKTICKS OR MARKDOWN FORMATTING. "
#         f"PROPERLY QUOTE THE KEYS AND VALUES IN THE DICTIONARY FOR EASIER PARSING WITH Python's ast.literal_eval."
#     )
#     response = model.generate_content([prompt, img])
#     print(response.text)
#     answers = []
#     try:
#         answers = ast.literal_eval(response.text)
#     except Exception as e:
#         print(f"Error in parsing response from Gemini API: {e}")
#     print('returned answer ', answers)
#     for answer in answers:
#         if 'assign' in answer:
#             answer['assign'] = True
#         else:
#             answer['assign'] = False
#     return answers
import google.generativeai as genai
import ast
import json
import re
from PIL import Image
from constants import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

def analyze_image(img: Image, dict_of_vars: dict):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    dict_of_vars_str = json.dumps(dict_of_vars, ensure_ascii=False)
    
    prompt = (
        f"You have been given an image with some mathematical expressions, equations, or graphical problems, and you need to solve them. "
        f"Note: Use the PEMDAS rule for solving mathematical expressions. PEMDAS stands for the Priority Order: Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right). Parentheses have the highest priority, followed by Exponents, then Multiplication and Division, and lastly Addition and Subtraction. "
        f"For example: "
        f"Q. 2 + 3 * 4 "
        f"(3 * 4) => 12, 2 + 12 = 14. "
        f"Q. 2 + 3 + 5 * 4 - 8 / 2 "
        f"5 * 4 => 20, 8 / 2 => 4, 2 + 3 => 5, 5 + 20 => 25, 25 - 4 => 21. "
        f"YOU CAN HAVE FIVE TYPES OF EQUATIONS/EXPRESSIONS IN THIS IMAGE, AND ONLY ONE CASE SHALL APPLY EVERY TIME: "
        f"Following are the cases: "
        f"1. Simple mathematical expressions like 2 + 2, 3 * 4, 5 / 6, 7 - 8, etc.: In this case, solve and return the answer in the format of a LIST OF ONE DICT [{{'expr': given expression, 'result': calculated answer}}]. "
        f"2. Set of Equations like x^2 + 2x + 1 = 0, 3y + 4x = 0, 5x^2 + 6y + 7 = 12, etc.: In this case, solve for the given variable, and the format should be a COMMA SEPARATED LIST OF DICTS, with dict 1 as {{'expr': 'x', 'result': 2, 'assign': True}} and dict 2 as {{'expr': 'y', 'result': 5, 'assign': True}}. This example assumes x was calculated as 2, and y as 5. Include as many dicts as there are variables. "
        f"3. Assigning values to variables like x = 4, y = 5, z = 6, etc.: In this case, assign values to variables and return another key in the dict called {{'assign': True}}, keeping the variable as 'expr' and the value as 'result' in the original dictionary. RETURN AS A LIST OF DICTS. "
        f"4. Analyzing Graphical Math problems, which are word problems represented in drawing form, such as cars colliding, trigonometric problems, problems on the Pythagorean theorem, adding runs from a cricket wagon wheel, etc. These will have a drawing representing some scenario and accompanying information with the image. PAY CLOSE ATTENTION TO DIFFERENT COLORS FOR THESE PROBLEMS. You need to return the answer in the format of a LIST OF ONE DICT [{{'expr': given expression, 'result': calculated answer}}]. "
        f"5. Detecting Abstract Concepts that a drawing might show, such as love, hate, jealousy, patriotism, or a historic reference to war, invention, discovery, quote, etc. USE THE SAME FORMAT AS OTHERS TO RETURN THE ANSWER, where 'expr' will be the explanation of the drawing, and 'result' will be the abstract concept. "
        f"Analyze the equation or expression in this image and return the answer according to the given rules: "
        f"Make sure to use extra backslashes for escape characters like \\f -> \\\\f, \\n -> \\\\n, etc. "
        f"Here is a dictionary of user-assigned variables. If the given expression has any of these variables, use its actual value from this dictionary accordingly: {dict_of_vars_str}. "
        f"DO NOT USE BACKTICKS OR MARKDOWN FORMATTING. "
        f"PROPERLY QUOTE THE KEYS AND VALUES IN THE DICTIONARY FOR EASIER PARSING WITH Python's ast.literal_eval."
    )
    
    try:
        print("Sending request to Gemini API...")
        response = model.generate_content([prompt, img])
        if not response or not hasattr(response, 'text'):
            raise ValueError("Invalid response from Gemini API")
        
        response_text = response.text
        print("Raw response from API:", response_text)
        
        # Clean up the response if it contains code blocks or extra formatting
        cleaned_response = clean_response_text(response_text)
        print("Cleaned response:", cleaned_response)
        
        # Try to parse the response
        answers = parse_response(cleaned_response)
        
        # Validate the answers
        validate_answers(answers)
        
        # Ensure 'assign' key is properly set
        for answer in answers:
            answer['assign'] = answer.get('assign', False)
        
        print("Processed answers:", answers)
        return answers
    
    except Exception as e:
        print(f"Error in response processing: {str(e)}")
        # Return a minimal valid response to prevent UnboundLocalError
        return []

def clean_response_text(text):
    """Remove code blocks, backticks and other formatting issues"""
    # Remove markdown code blocks (```python ... ```)
    text = re.sub(r'```(?:python)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
    
    # Remove single backticks
    text = text.replace('`', '')
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    return text

def parse_response(text):
    """Attempt different parsing strategies for the response"""
    try:
        # First try direct parsing
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        # If that fails, try to extract just the list part
        list_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
        if list_match:
            try:
                return ast.literal_eval(f"[{list_match.group(1)}]")
            except (SyntaxError, ValueError):
                pass
        
        # If all else fails, try to construct a minimal valid response
        print("Failed to parse response, returning empty list")
        return []

def validate_answers(answers):
    """Ensure answers meet expected format"""
    if not isinstance(answers, list):
        raise ValueError("Response is not a list")
    
    for idx, answer in enumerate(answers):
        if not isinstance(answer, dict):
            raise ValueError(f"Item at index {idx} is not a dictionary")
        
        if 'expr' not in answer:
            raise ValueError(f"Item at index {idx} missing 'expr' key")
        
        if 'result' not in answer:
            raise ValueError(f"Item at index {idx} missing 'result' key")
        
    return True