import cv2
import numpy as np
import service.main as s
# def malaria_detector(img_array):
#     # --- Handle different image formats ---
#     if len(img_array.shape) == 2:  # grayscale → RGB
#         img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
#     elif img_array.shape[2] == 4:  # RGBA → RGB
#         img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
#     else:  # BGR → RGB (normal OpenCV read)
#         img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    

#     # --- Preprocess image ---
#     test_image = cv2.resize(img_array, (224, 224))
#     test_image = np.float32(test_image) 
#     test_image = np.expand_dims(test_image, 0)    # add batch dimension

#     # --- Define class labels ---
#     classes = ['Uninfected', 'Parasitized']

#     # --- Run inference ---
#     input_name = s.m_q.get_inputs()[0].name       # safer than hardcoding "input_layer"
#     output_name = s.m_q.get_outputs()[0].name     # safer than hardcoding "output_0"

#     onnx_pred = s.m_q.run([output_name], {input_name: test_image})
#     probs = onnx_pred[0][0]  # model output
#     print(onnx_pred)
#     # --- Postprocess predictions ---
#     probs = probs.tolist()
#     # If model outputs a single probability → make it [p, 1-p]
#     probs = [probs[0], 1 - probs[0]]

#     class_idx = int(np.argmax(probs))
#     class_name = classes[class_idx]

#     # --- Return JSON-friendly result ---
#     return {
#         "prediction": class_name,
#         "class_index": class_idx,
#         "probabilities": probs
#     }
def malaria_detector(img_array):
    # --- Handle different image formats ---
    if len(img_array.shape) == 2:  # grayscale → RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA → RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:  # BGR → RGB (normal OpenCV read)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    

    # --- Preprocess image ---
    test_image = cv2.resize(img_array, (224, 224))
    test_image = np.float32(test_image) 
    test_image = np.expand_dims(test_image, 0)    # add batch dimension

    # --- Define class labels ---
    classes = ['Uninfected', 'Parasitized']

    # --- Run inference ---
    input_name = s.m_q.get_inputs()[0].name       # safer than hardcoding "input_layer"
    output_name = s.m_q.get_outputs()[0].name     # safer than hardcoding "output_0"

    onnx_pred = s.m_q.run([output_name], {input_name: test_image})
    probs = onnx_pred[0][0]  # model output
    print(onnx_pred)
    
    # --- Postprocess predictions ---
    probs = probs.tolist()
    # If model outputs a single probability → make it [p, 1-p]
    probs = [probs[0], 1 - probs[0]]

    # --- Apply threshold ---
    threshold = 0.672
    # probs[0] is "Uninfected" probability
    if probs[0] >= threshold:
        class_idx = 0  # Uninfected
    else:
        class_idx = 1  # Parasitized
    
    class_name = classes[class_idx]

    # --- Return JSON-friendly result ---
    return {
        "prediction": class_name,
        "class_index": class_idx,
        "probabilities": probs
    }

