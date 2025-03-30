# from fastapi import APIRouter
# import base64
# from io import BytesIO
# from apps.calculator.utils import analyze_image
# from schema import ImageData
# from PIL import Image

# router = APIRouter()

# @router.post('')
# async def run(data: ImageData):
#     image_data = base64.b64decode(data.image.split(",")[1])  # Assumes data:image/png;base64,<data>
#     image_bytes = BytesIO(image_data)
#     image = Image.open(image_bytes)
#     responses = analyze_image(image, dict_of_vars=data.dict_of_vars)
#     print(responses)
#     data = []
#     for response in responses:
#         data.append(response)
#     print('response in route: ', response)
#     return {"message": "Image processed", "data": data, "status": "success"}
from fastapi import APIRouter, HTTPException
import base64
from io import BytesIO
from apps.calculator.utils import analyze_image
from schema import ImageData
from PIL import Image
from typing import Dict, Any, List

router = APIRouter()

@router.post('')
async def run(data: ImageData):
    try:
        # Decode base64 image
        try:
            # Split by comma to handle data URI format (data:image/png;base64,<data>)
            image_parts = data.image.split(",", 1)
            image_data = base64.b64decode(image_parts[1] if len(image_parts) > 1 else image_parts[0])
            image_bytes = BytesIO(image_data)
            image = Image.open(image_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        print(image , data.dict_of_vars)
        # Process image with variables
        responses = analyze_image(image, dict_of_vars=data.dict_of_vars)
        
        # Ensure responses is a list, even if empty
        if responses is None:
            responses = []
        
        # Create a separate result list to avoid reference issues
        result_data: List[Dict[str, Any]] = []
        for response in responses:
            result_data.append(response)
        
        # Log the complete result data, not just the last response
        print('responses in route: ', result_data)
        
        # Return a properly structured response
        return {
            "message": "Image processed",
            "data": result_data,
            "status": "success"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        # Catch and convert any other exceptions to HTTP 500
        error_detail = f"Image processing failed: {str(e)}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)